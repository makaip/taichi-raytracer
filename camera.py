import taichi as ti
import taichi.math as tm

from manifold import Manifold

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


@ti.dataclass
class Camera:
    pos: vec3
    rot: vec3

    image_width: int
    image_height: int

    fov: float

    p00: vec3
    pdu: vec3
    pdv: vec3

    dfu: vec3 = vec3(1, 0, 0)
    dfv: vec3 = vec3(0, 1, 0)
    dfw: vec3 = vec3(0, 0, 1)

    @ti.func
    def rotate(self, v: vec3) -> vec3:
        cx = tm.cos(self.rot.x), sx = tm.sin(self.rot.x)
        cy = tm.cos(self.rot.y), sy = tm.sin(self.rot.y)
        cz = tm.cos(self.rot.z), sz = tm.sin(self.rot.z)

        R = mat3(
            {cy * cz,   sx * sy * cz - cx * sz,     cx * sy * cz + sx * sz},
            {cy * sz,   sx * sy * sz + cx * cz,     cx * sy * sz - sx * cz},
            {sy,        sx * cy,                    cx * cy}
        )

        out = vec3(0)

        for i in range(3):
            out[i] = R[i][0] * v.x + R[i][1] * v.y + R[i][2] * v.z
        
        return out

    @ti.func
    def init(self):
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
    
    def render(self, manifold, scene):
        self.init()
        basis = manifold.basis(self.pos)
        g = manifold.metric_tensor(self.pos, basis)

        # self.render_kernel(manifold, scene, g)

    # @ti.kernel
    # def render_kernel(self, manifold: Manifold, scene: list, g: mat3x4):
        