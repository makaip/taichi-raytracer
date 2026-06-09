import taichi as ti

vec3 = ti.types.vector(3, float)

@ti.dataclass
class Ray:
    origin: vec3
    direction: vec3

    @ti.func
    def at(self, t: float):
        return self.origin + self.direction * t


@ti.dataclass
class Interval:
    min: float
    max: float

    @ti.func
    def size(self) -> float:
        return self.max - self.min
    
    @ti.func
    def contains(self, x: float) -> bool:
        return self.min <= x and x <= self.max

    @ti.func
    def surrounds(self, x: float) -> bool:
        return self.min < x and x < self.max


def rotate_about_z(vec: vec3, angle: float) -> vec3:
    c, s = ti.math.cos(angle), ti.math.sin(angle)
    rot_matrix = ti.math.mat3([
        [c,  -s,   0.0],
        [s,   c,   0.0],
        [0.0, 0.0, 1.0]
    ])
    
    return rot_matrix @ vec


def rotate_about_y(vec: vec3, angle: float) -> vec3:
    c, s = ti.math.cos(angle), ti.math.sin(angle)
    rot_matrix = ti.math.mat3([
        [c,   0.0, s],
        [0.0, 1.0, 0.0],
        [-s,  0.0, c]
    ])
    
    return rot_matrix @ vec


empty_interval = Interval(ti.math.inf, -ti.math.inf)
universe_interval = Interval(-ti.math.inf, ti.math.inf)