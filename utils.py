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

empty_interval = Interval(ti.math.inf, -ti.math.inf)
universe_interval = Interval(-ti.math.inf, ti.math.inf)