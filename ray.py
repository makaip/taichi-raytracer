import taichi as ti
import taichi.math as tm

from geom import *

vec3 = ti.types.vector(3, float)

MAX_STEPS = 100

@ti.dataclass
class Ray:
    origin: vec3
    direction: vec3

    @ti.func
    def at(
        self, 
        t: float, 
        k: float
    ):
        return exp_map(self.origin, self.direction * t, k)


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


empty_interval = Interval(tm.inf, -tm.inf)
universe_interval = Interval(-tm.inf, tm.inf)
