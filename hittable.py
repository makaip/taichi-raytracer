
from abc import abstractmethod
import taichi as ti
import numpy as np

from ray import Ray

vec3 = ti.types.vector(3, float)


@ti.dataclass
class HitRecord:
    p: vec3
    normal: vec3
    t: float
    front_face: bool

    @ti.func
    def set_face_normal(self, ray: Ray, outward_normal: vec3):
        self.front_face = ti.math.dot(ray.direction, outward_normal) < 0
        if self.front_face:
            self.normal = outward_normal
        else:
            self.normal = -outward_normal


@ti.data_oriented
class Hittable:
    def __init__(
            self,
            ray: Ray,
            tmin: float,
            tmax: float,
            rec: HitRecord
    ):
        self.ray = ray
        self.tmin = tmin
        self.tmax = tmax
        self.rec = rec

    @ti.func
    def hit(
        self,
        origin: vec3,
        direction: vec3,
        tmin: float,
        tmax: float,
        rec: HitRecord
    ) -> float:
        return 0


@ti.data_oriented
class HittableList(Hittable):
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
            self.objects[idx].center = object.center
            self.objects[idx].radius = object.radius
            self.count[None] += 1
        else:
            raise OverflowError("HittableList is full")

    def hit(
        self,
        ray: Ray,
        tmin: float,
        tmax: float,
        rec: HitRecord
    ) -> bool:
        hit = False
        closest = tmax

        final_rec = HitRecord(
            p=vec3(0),
            normal=vec3(0),
            t=0.0,
            front_face=False
        )

        for i in range(self.count[None]):
            is_hit, temp_rec = self.objects[i].hit(ray, tmin, closest)

            if is_hit:
                hit = True
                closest = temp_rec.t
                final_rec = temp_rec

        return hit, final_rec


@ti.dataclass
class Sphere(Hittable):
    def __init__(self, center: vec3, radius: float):
        self.center = center
        self.radius = radius

    def hit(
        self,
        ray: Ray,
        tmin: float,
        tmax: float,
        rec: HitRecord
    ) -> bool:
        oc = self.center - ray.origin

        a = ray.direction.norm_sqr()
        h = ti.math.dot(ray.direction, oc)
        c = oc.norm_sqr() - (self.radius ** 2)

        disc = (h ** 2) - (a * c)

        result = None

        if (disc < 0):
            result = False
        else:
            sqrtd = ti.sqrt(disc)
            root = (h - sqrtd) / a

            if (root <= tmin or tmax <= root):
                root = (h + sqrtd) / a

                if (root <= tmin or tmax <= root):
                    result = False
                else:
                    result = True
            else:
                result = True

        if result is True:
            rec.t = root
            rec.p = ray.at(rec.t)
            outward_normal = (rec.p - self.center) / self.radius
            rec.set_face_normal(ray, outward_normal)

        return result
