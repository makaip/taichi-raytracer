import taichi as ti
import taichi.math as tm

from manifold import Manifold

vec3 = ti.types.vector(3, float)
vec4 = ti.types.vector(4, float)

mat3 = ti.types.matrix(3, 3, float)

@ti.data_oriented
class Scene:
    def __init__(self, max_objects=100):
        self.objects = Sphere.field(shape=max_objects)
        self.count = ti.field(
            dtype=ti.i32,
            shape=()
        )
        self.count[None] = 0

    def clear(self):
        self.count[None] = 0

    def add(self, object):
        idx = self.count[None]
        if idx < self.objects.shape[0]:
            self.objects[idx].origin = object.origin
            self.objects[idx].radius = object.radius
            self.count[None] += 1
        else:
            raise OverflowError("HittableList is full")

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

        raised = vec3(0.0)
        for i in range(3):
            for j in range(3):
                raised[i] += g_inv[i,j] * grad[j]
        
        return tm.normalize(raised)
