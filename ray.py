import taichi as ti

vec3 = ti.types.vector(3, float)

@ti.dataclass
class Ray:
    origin: vec3
    direction: vec3

    @ti.func
    def at(self, t: float):
        return self.origin + self.direction * t

# ray = Ray(vec3(0), vec3(1, 1, 0))

# @ti.kernel
# def bruh():
#     print(at(ray, 0.5))

# bruh()