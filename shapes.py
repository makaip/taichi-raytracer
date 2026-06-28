import taichi as ti
import taichi.math as tm

from manifold import Manifold

vec3 = ti.types.vector(3, float)
vec4 = ti.types.vector(4, float)

mat3 = ti.types.matrix(3, 3, float)


@ti.dataclass
class Sphere:
    origin: vec4
    radius: float
    
    @ti.func
    def sdf(self, pos: vec3, manifold: ti.template()):
        pos4d = manifold.f(pos.x, pos.y, pos.z)
        diff = pos4d - self.origin
        dist = tm.sqrt(tm.dot(diff, diff))
        
        return dist - self.radius

    @ti.func
    def sdf_normal(self, pos: vec3, g_inv: mat3, m: ti.template(), h: float = 1e-4):
        dx = self.sdf(vec3(pos.x + h, pos.y, pos.z), m) - self.sdf(vec3(pos.x - h, pos.y, pos.z), m)
        dy = self.sdf(vec3(pos.x, pos.y + h, pos.z), m) - self.sdf(vec3(pos.x, pos.y - h, pos.z), m)
        dz = self.sdf(vec3(pos.x, pos.y, pos.z + h), m) - self.sdf(vec3(pos.x, pos.y, pos.z - h), m)

        grad = tm.normalize(vec3(dx, dy, dz))

        raised = vec3(0)
        for i in range(3):
            for j in range(3):
                raised[i] += g_inv[i][j] * grad[j]
        
        return tm.normalize(raised)
