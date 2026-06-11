import taichi as ti
import taichi.math as tm

import numpy as np

from utils import *
from ray import *
from geom import *

HIT_EPS = 1e-3
MAX_DIST = 100

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
        ray_t: Interval,
        k: float
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
            is_hit, temp_rec = self.objects[i].hit(ray, Interval(ray_t.min, closest), k)

            if is_hit:
                hit = True
                closest = temp_rec.t
                final_rec = temp_rec

        return hit, final_rec


# https://typhomnt.github.io/teaching/ray_tracing/raymarching_intro/
@ti.dataclass
class Sphere():
    center: vec3
    radius: float

    @ti.func
    def hit(
        self,
        ray: Ray,
        ray_t: Interval,
        k: float
    ) -> tuple:
        p = ray.origin
        v = ray.direction.normalized()
        t = 0.0
        hit = False

        rec = HitRecord(vec3(0), vec3(0), 0.0, False)

        for _ in range(MAX_STEPS):
            dist = self.sdf(p, k)
            if dist < HIT_EPS and ray_t.surrounds(t):
                hit = True
                rec.t = t
                rec.p = p

                outward = self.sdf_grad(p, k)
                rec.set_face_normal(ray, outward)
                break
            
            if t > MAX_DIST:
                break
            
            p, v = rk4_step(p, v, dist * 0.8, k)
            t += dist * 0.8
        
        return hit, rec


    @ti.func
    def sdf(
        self,
        p: vec3,
        k: float
    ) -> float:
        return geodesic_dist(p, self.center, k) - self.radius

    @ti.func
    def sdf_grad(
        self,
        p: vec3,
        k: float
    ) -> vec3:
        eps = 1e-3
        
        eps_x = vec3(eps, 0, 0)
        eps_y = vec3(0, eps, 0)
        eps_z = vec3(0, 0, eps)

        norm_x = self.sdf(exp_map(p,  eps_x, k), k) - self.sdf(exp_map(p, -eps_x, k), k)
        norm_y = self.sdf(exp_map(p,  eps_y, k), k) - self.sdf(exp_map(p, -eps_y, k), k)
        norm_z = self.sdf(exp_map(p,  eps_z, k), k) - self.sdf(exp_map(p, -eps_z, k), k)

        return vec3(norm_x, norm_y, norm_z).normalized()
