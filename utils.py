import taichi as ti
import taichi.math as tm


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

    @ti.func
    def clamp(self, x: float) -> float:
        result = x
        if (x < self.min): result = self.min
        if (self.max < x): result = self.max
        return result


@ti.func
def sample_square():
    return vec3(
        ti.random(dtype=float) - 0.5, 
        ti.random(dtype=float) - 0.5, 
        0.0
    )


@ti.func
def random_vector(
    r_min: float = 0.0,
    r_max: float = 1.0
):
    return vec3(
        (ti.random(dtype=float) * (r_max - r_min)) + r_min,
        (ti.random(dtype=float) * (r_max - r_min)) + r_min,
        (ti.random(dtype=float) * (r_max - r_min)) + r_min
    )


@ti.func
def random_on_hemi(normal: vec3):
    on_unit_sphere = random_vector(-1.0, 1.0).normalized()
    result = -on_unit_sphere

    if tm.dot(on_unit_sphere, normal) > 0.0:
        result = on_unit_sphere
    
    return result

@ti.func
def linear_to_gamma(lin_comp):
    result = 0
    
    if lin_comp > 0:
        result =  tm.sqrt(lin_comp)

    return result


def rotate_about_z(vec: vec3, angle: float) -> vec3:
    c, s = tm.cos(angle), tm.sin(angle)
    rot_matrix = tm.mat3([
        [c,  -s,   0.0],
        [s,   c,   0.0],
        [0.0, 0.0, 1.0]
    ])
    
    return rot_matrix @ vec


def rotate_about_y(vec: vec3, angle: float) -> vec3:
    c, s = tm.cos(angle), tm.sin(angle)
    rot_matrix = tm.mat3([
        [c,   0.0, s],
        [0.0, 1.0, 0.0],
        [-s,  0.0, c]
    ])
    
    return rot_matrix @ vec


empty_interval = Interval(tm.inf, -tm.inf)
universe_interval = Interval(-tm.inf, tm.inf)
