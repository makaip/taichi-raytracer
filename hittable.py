import taichi as ti
import taichi.math as tm

import numpy as np

from utils import *

vec3 = ti.types.vector(3, float)

@ti.dataclass
class HitRecord:
    p: vec3
    normal: vec3
    t: float
    front_face: bool

    @ti.func
    def set_face_normal(self, ray: Ray, outward_normal: vec3):
        self.front_face = tm.dot(ray.direction, outward_normal) < 0
        if self.front_face:
            self.normal = outward_normal
        else:
            self.normal = -outward_normal


@ti.data_oriented
class HittableList():
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

    @ti.func
    def hit(
        self,
        ray: Ray,
        ray_t: Interval
    ) -> tuple:
        hit = False
        closest = ray_t.max

        final_rec = HitRecord(
            p=vec3(0),
            normal=vec3(0),
            t=0.0,
            front_face=False
        )

        for i in range(self.count[None]):
            is_hit, temp_rec = self.objects[i].hit(ray, Interval(ray_t.min, closest))

            if is_hit:
                hit = True
                closest = temp_rec.t
                final_rec = temp_rec

        return hit, final_rec

# hittable
@ti.dataclass
class Sphere():
    center: vec3
    radius: float

    @ti.func
    def hit(
        self,
        ray: Ray,
        ray_t: Interval
    ) -> tuple:
        oc = self.center - ray.origin

        a = ray.direction.norm_sqr()
        h = tm.dot(ray.direction, oc)
        c = oc.norm_sqr() - (self.radius ** 2)

        disc = (h ** 2) - (a * c)

        is_hit = False
        rec = HitRecord(p=vec3(0), normal=vec3(0), t=0.0, front_face=False)

        if disc >= 0:
            sqrtd = ti.sqrt(disc)
            root = (h - sqrtd) / a

            if not ray_t.surrounds(root):
                root = (h + sqrtd) / a

            if ray_t.surrounds(root):
                is_hit = True
                rec.t = root
                rec.p = ray.at(rec.t)
                outward_normal = (rec.p - self.center) / self.radius
                rec.set_face_normal(ray, outward_normal)

        return is_hit, rec

