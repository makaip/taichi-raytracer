import taichi as ti

vec3 = ti.types.vector(3, float)
vec4 = ti.types.vector(4, float)

@ti.dataclass
class Plane:
    origin: vec4
    normal: vec4
    manifold: None

    def __init__(self, origin: vec3, normal: vec3, manifold):
        self.manifold = manifold
