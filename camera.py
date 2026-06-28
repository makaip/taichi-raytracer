import taichi as ti
import taichi.math as tm
import math

from manifold import Manifold

vec3 = ti.types.vector(3, float)
vec4 = ti.types.vector(4, float)

mat3 = ti.types.matrix(3, 3, float)
mat3x4 = ti.types.matrix(3, 4, float)

STEP_SIZE = 1e-2
HIT_DIST = 1e-2
MAX_DEPTH = 500


@ti.data_oriented
class Camera:
    pixels: ti.Vector.field

    def __init__(self, pos: vec3, pitch: float, yaw: float, image_width: int, image_height: int, fov: float):
        self.pos = pos
        self.rot = ti.Vector([0.0, 0.0, 0.0])

        self.pitch = pitch
        self.yaw = yaw

        self.image_width = image_width
        self.image_height = image_height
        self.fov = fov

        self.vup = ti.Vector([0, 1, 0])

        th = self.fov * (math.pi / 180)
        h = math.tan(th / 2)

        self.vh = 2 * h
        self.vw = self.vh * (self.image_width / self.image_height)

        self.update()

        self.pixels = ti.Vector.field(
            n=3,
            dtype=ti.f32,
            shape=(self.image_width, self.image_height)
        )

    def update(self):
        self.rot = ti.Vector([
            math.sin(self.yaw) * math.cos(self.pitch),
            math.sin(self.pitch),
            -math.cos(self.yaw) * math.cos(self.pitch)
        ]).normalized()

        self.yaw_basis = self.vup.cross(self.rot).normalized()
        self.pitch_basis = self.rot.cross(self.yaw_basis).normalized()

        vu = self.vw * self.yaw_basis
        vv = -self.vh * self.pitch_basis

        self.pdu = vu / self.image_width
        self.pdv = vv / self.image_height

        vul = self.pos - self.rot - (vu / 2) - (vv / 2)
        self.p00 = vul + 0.5 * (self.pdu + self.pdv)

    def render(self, manifold, scene):
        self.update()
        self.render_kernel(manifold, scene, self.pos, self.p00, self.pdu, self.pdv)

    @ti.kernel
    def render_kernel(
        self, 
        manifold: ti.template(),
        scene: ti.template(),

        pos: vec3,
        p00: vec3,
        pdu: vec3,
        pdv: vec3
    ):
        basis = manifold.basis(pos)
        g = manifold.metric_tensor(pos, basis)

        for x, y in self.pixels:
            col_r = 1; col_g = 1; col_b = 1

            rpos = p00 + (x * pdu) + (y * pdv)
            rdir = rpos - pos

            mns = 0.0
            for i in range(3):
                for j in range(3):
                    mns += rdir[i] * g[i,j] * rdir[j]

            scale = 1 / tm.sqrt(mns)
            rdir *= scale

            hit, norm, depth = self.march_ray(manifold, scene, rpos, rdir)
            depth_col = MAX_DEPTH / ((4 * depth) + MAX_DEPTH)

            if hit:
                col_r = tm.clamp((norm.x * 0.5 + 0.5) * 255 * depth_col, 0, 255)
                col_g = tm.clamp((norm.y * 0.5 + 0.5) * 255 * depth_col, 0, 255)
                col_b = tm.clamp((norm.z * 0.5 + 0.5) * 255 * depth_col, 0, 255)

            self.pixels[x, y] = vec3(col_r, col_g, col_b)

    @ti.func
    def march_ray(self, manifold: ti.template(), scene: ti.template(), rpos: vec3, rdir: vec3):
        hit = False
        norm = vec3(0)

        tmp_pos = rpos
        tmp_dir = rdir
        steps = 0

        while steps < MAX_DEPTH:
            min_dist = HIT_DIST
            closest_idx = -1

            for idx in range(scene.count[None]):
                dist = scene.objects[idx].sdf(tmp_pos, manifold)

                if (dist < min_dist):
                    min_dist = dist
                    closest_idx = idx
                
            if (min_dist < HIT_DIST):
                basis = manifold.basis(tmp_pos)
                g = manifold.metric_tensor(tmp_pos, basis)
                norm = scene.objects[closest_idx].sdf_normal(tmp_pos, tm.inverse(g), manifold)

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
