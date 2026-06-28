import taichi as ti
import taichi.math as tm

from manifold import Manifold
from globals import scene, pixels, IMAGE_WIDTH, IMAGE_HEIGHT

vec3 = ti.types.vector(3, float)
vec4 = ti.types.vector(4, float)

mat3 = ti.types.matrix(3, 3, float)
mat3x4 = ti.types.matrix(3, 4, float)

STEP_SIZE = 1e-2
HIT_DIST = 1e-2
MAX_DEPTH = 500


@ti.dataclass
class Ray:
    pos: vec3
    dir: vec3


@ti.data_oriented
class Camera:
    pos: vec3
    rot: vec3

    image_width: int
    image_height: int

    fov: float

    p00: vec3
    pdu: vec3
    pdv: vec3

    dfu: vec3
    dfv: vec3
    dfw: vec3

    def __init__(self, pos: vec3, rot: vec3, image_width: int, image_height: int, fov: float):
        self.pos = pos
        self.rot = rot

        self.image_width = image_width
        self.image_height = image_height
        self.fov = fov

    def rotate(self, v: vec3) -> vec3:
        cx = tm.cos(self.rot.x); sx = tm.sin(self.rot.x)
        cy = tm.cos(self.rot.y); sy = tm.sin(self.rot.y)
        cz = tm.cos(self.rot.z); sz = tm.sin(self.rot.z)

        R = mat3(
            cy * cz,   sx * sy * cz - cx * sz,     cx * sy * cz + sx * sz,
            cy * sz,   sx * sy * sz + cx * cz,     cx * sy * sz - sx * cz,
            sy,        sx * cy,                    cx * cy
        )

        return R @ v


    def init(self):
        self.dfu = vec3(1, 0, 0)
        self.dfv = vec3(0, 1, 0)
        self.dfw = vec3(0, 0, 1)

        th = self.fov * (tm.pi / 180)
        h = tm.tan(th / 2)

        vh = 2 * h
        vw = vh * (self.image_width / self.image_height)

        bu = self.rotate(self.dfu)
        bv = self.rotate(self.dfv)
        bw = self.rotate(self.dfw)

        vu = vw * bu
        vv = -vh * bv

        pdu = vu / self.image_width
        pdv = vv / self.image_height

        vul = self.pos - bw - (vu / 2) - (vv / 2)
        self.p00 = vul + 0.5 * (pdu + pdv)

    def render(self, manifold):
        self.init()
        self.render_kernel(manifold)

    @ti.kernel
    def render_kernel(self, manifold: ti.template()):
        basis = manifold.basis(self.pos)
        g = manifold.metric_tensor(self.pos, basis)

        for y in range(self.image_height):
            for x in range(self.image_width):
                col_r = 1; col_g = 1; col_b = 1

                rpos = self.p00 + (x * self.pdu) + (y * self.pdv)
                rdir = rpos - self.pos

                mns = 0
                for i in range(3):
                    for j in range(3):
                        mns += rdir[i] * g[i][j] * rdir[j]

                scale = 1 / tm.sqrt(mns)
                rdir *= scale

                hit, norm, depth = self.march_ray(manifold, rpos, rdir)
                depth_col = MAX_DEPTH / ((4 * depth) + MAX_DEPTH)

                if hit:
                    col_r = tm.clamp((norm.x * 0.5 + 0.5) * 255 * depth_col, 0, 255)
                    col_g = tm.clamp((norm.y * 0.5 + 0.5) * 255 * depth_col, 0, 255)
                    col_b = tm.clamp((norm.z * 0.5 + 0.5) * 255 * depth_col, 0, 255)

                pixels[x, y] = vec3(col_r, col_g, col_b)

    @ti.func
    def march_ray(self, manifold: ti.template(), rpos: vec3, rdir: vec3):
        hit = False
        norm = vec3(0)

        tmp_pos = rpos
        tmp_dir = rdir
        steps = 0

        while steps < MAX_DEPTH:
            min_dist = HIT_DIST
            closest_idx = -1

            for idx in range(scene.shape[0]):
                dist = scene[idx].sdf(tmp_pos, manifold)

                if (dist < min_dist):
                    min_dist = dist
                    closest_idx = idx
                
            if (min_dist < HIT_DIST):
                basis = manifold.basis(tmp_pos)
                g = manifold.metric_tensor(tmp_pos, basis)
                norm = scene[closest_idx].sdf_normal(tmp_pos, tm.inverse(g))

                hit = True
                break
            
            dt = tm.min(min_dist * 0.8, 0.4)

            if (dt < 1e-4):
                dt = 1e-4
            
            new_pos, new_dir = manifold.step(tmp_pos, tmp_dir, dt)

            tmp_pos = new_pos
            tmp_dir = new_dir
            steps += 1
        
        return hit, norm, steps
