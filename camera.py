import math
import numpy as np

import taichi as ti
import taichi.math as tm

from hittable import *
from utils import rotate_about_y, rotate_about_z

vec3 = ti.types.vector(3, float)

@ti.data_oriented
class Camera:
    pixels: ti.Vector.field

    def __init__(
            self,
            samples_per_pixel: int,
            image_width: int,
            image_height: int,
            vfov: float,
            max_depth: int,
            gamma: float
    ):
        self.speed = 0.1
        self.kappa = 0
        
        self.samples_per_pixel = samples_per_pixel

        self.image_width = image_width
        self.image_height = image_height

        self.vfov = vfov

        self.max_depth = max_depth
    
        self.gamma = gamma
        self.focal_length = 1.0

        # use ti.Vector outside @ti scope
        self.vup = ti.Vector([0.0, 1.0, 0.0])

        self.position = ti.Vector([0.0, 0.0, 0.0])
        self.rotation = ti.Vector([0.0, 0.0, 0.0])      # camera normal (facing fowards)

        # ___________________________________________

        theta = math.radians(vfov)
        h = math.tan(theta / 2)
        self.view_height = 2 * h * self.focal_length
        self.view_width = self.view_height * (self.image_width / self.image_height)

        self.yaw = math.radians(180)
        self.pitch = 0.0

        self.update_view()
            
        self.pixels = ti.Vector.field(
            n=3,
            dtype=float,
            shape=(
                self.image_width,
                self.image_height
            )
        )
    
    def update_view(self):
        self.rotation = ti.Vector([
            math.sin(self.yaw) * math.cos(self.pitch),
            math.sin(self.pitch),
            -math.cos(self.yaw) * math.cos(self.pitch)
        ]).normalized()

        self.yaw_basis = self.vup.cross(self.rotation).normalized()
        self.pitch_basis = self.rotation.cross(self.yaw_basis).normalized()

        self.view_u = self.view_width * self.yaw_basis
        self.view_v = self.view_height * self.pitch_basis

        self.pixel_delta_u = self.view_u / self.image_width
        self.pixel_delta_v = self.view_v / self.image_height

        self.view_upper_left = self.position - \
            (self.focal_length * self.rotation) - \
            (self.view_u / 2.0) - (self.view_v / 2.0)
            
        self.init_pixel_loc = self.view_upper_left + \
            0.5 * (self.pixel_delta_u + self.pixel_delta_v)
    
    def handle_motion(self, gui: ti.GUI):
        # navigation
        if gui.is_pressed('w'):
            self.position -= self.rotation * self.speed
        if gui.is_pressed('s'):
            self.position += self.rotation * self.speed
        if gui.is_pressed('a'):
            self.position -= self.yaw_basis * self.speed
        if gui.is_pressed('d'):
            self.position += self.yaw_basis * self.speed
        if gui.is_pressed('e'):
            self.position += self.vup * self.speed
        if gui.is_pressed('q'):
            self.position -= self.vup * self.speed
        
        # rotation
        if gui.is_pressed(ti.GUI.LEFT):
            self.yaw -= 0.05
        if gui.is_pressed(ti.GUI.RIGHT):
            self.yaw += 0.05
        if gui.is_pressed(ti.GUI.UP):
            self.pitch -= 0.05
            self.pitch = max(-math.radians(89), min(math.radians(89), self.pitch))
        if gui.is_pressed(ti.GUI.DOWN):
            self.pitch += 0.05
            self.pitch = max(-math.radians(89), min(math.radians(89), self.pitch))

        if gui.is_pressed('o'):
            self.kappa -= 0.005
            print(f"{self.kappa}\r")
        if gui.is_pressed('p'):
            self.kappa += 0.005
            print(f"{self.kappa}\r")
        
        self.update_view()

    @ti.func
    def ray_color(
            self,
            ray: Ray,
            depth: int,
            world: ti.template(),
            k: float
    ) -> vec3:
        color = vec3(0.0, 0.0, 0.0)
        attenuation = vec3(1.0, 1.0, 1.0)
        current_ray = ray
        brightness = 0.5

        for _ in range(depth):
            is_hit, rec = world.hit(current_ray, Interval(0.001, tm.inf), k)

            if is_hit:
                direction = random_on_hemi(rec.normal)
                current_ray = Ray(rec.p, direction)
                attenuation *= 1 - brightness
            else:
                unit_direction = current_ray.direction.normalized()
                a = 0.5 * (unit_direction[1] + 1.0)
                bg_color = (1.0 - a) * vec3(1.0, 1.0, 1.0) + a * vec3(0.5, 0.7, 1.0)

                color = attenuation * bg_color
                break

        return color
    
    @ti.func
    def get_ray(
        self,
        i: int,
        j: int,

        position: vec3,
        init_pixel_loc: vec3,
        pixel_delta_u: vec3,
        pixel_delta_v: vec3,
    ) -> Ray:
        offset = sample_square()
        pixel_center = init_pixel_loc + \
            ((i + offset[0]) * pixel_delta_u) + \
            ((j + offset[1]) * pixel_delta_v)
        
        ray_dir = (pixel_center - position).normalized()

        return Ray(position, ray_dir)

    # kernel treats globals as constants
    # so we pass any live params as args through render()
    @ti.kernel
    def render_kernel(
        self, 
        world: ti.template(),
        
        position: vec3,
        init_pixel_loc: vec3,
        pixel_delta_u: vec3,
        pixel_delta_v: vec3,
        k: float
    ):
        for i, j in self.pixels:
            self.pixels[i,j] = vec3(0, 0, 0)

            for sample in range(self.samples_per_pixel):
                ray = self.get_ray(
                    i, j,
                    position, init_pixel_loc,
                    pixel_delta_u, pixel_delta_v
                )
                self.pixels[i, j] += self.ray_color(ray, self.max_depth, world, k) * (1 / self.samples_per_pixel)
            
            self.pixels[i, j] = tm.clamp(self.pixels[i, j], 0, 1)
            self.pixels[i, j] = self.pixels[i, j] ** (1.0 / self.gamma)
    
    def render(
        self,
        world: ti.template(),
        k: float
    ):
        self.render_kernel(
            world,
            self.position,
            self.init_pixel_loc,
            self.pixel_delta_u,
            self.pixel_delta_v,
            k
        )
