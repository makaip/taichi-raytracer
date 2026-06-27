import taichi as ti
import taichi.math as tm

vec3 = ti.types.vector(3, float)
vec4 = ti.types.vector(4, float)

mat3 = ti.types.matrix(3, float)

@ti.dataclass
class Manifold:
    h: float

    @ti.func
    def f(self, u: float, v: float, w: float) -> vec4:
        return vec4(
            ti.cosh(u),
            ti.sinh(u) * ti.cos(v),
            ti.sinh(u) * ti.sin(v) * ti.cos(w),
            ti.sinh(u) * ti.sin(v) * ti.sin(w),
        )

    @ti.func
    def basis(self, pos: vec3) -> list:
        r_u = (self.f(pos.x + self.h, pos.y, pos.z) - self.f(pos.x - self.h, pos.y, pos.z)) / (2 * self.h)
        r_v = (self.f(pos.x, pos.y + self.h, pos.z) - self.f(pos.x, pos.y - self.h, pos.z)) / (2 * self.h)
        r_w = (self.f(pos.x, pos.y, pos.z + self.h) - self.f(pos.x, pos.y, pos.z - self.h)) / (2 * self.h)

        return [r_u, r_v, r_w]
    
    
