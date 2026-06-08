from PIL import Image
import taichi as ti
import numpy as np
import os

from typing import TextIO

from ray import *
from hittable import *

vec3 = ti.types.vector(3, float)

image_width, image_height = 800, 450
pixels = None

focal_length = 1.0

view_height = 2
view_width = view_height * (image_width / image_height)
camera_center = ti.Vector([0.0, 0.0, 0.0]) # use ti.Vector outside @ti scope

view_u = ti.Vector([view_width, 0.0, 0.0])
view_v = ti.Vector([0.0, -view_height, 0.0])

pixel_delta_u = view_u / image_width
pixel_delta_v = view_v / image_height

view_upper_left = camera_center - ti.Vector([0.0, 0.0, focal_length]) - (view_u / 2.0) - (view_v / 2.0)
init_pixel_loc = view_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v)

@ti.func
def ray_color(
    ray: Ray, 
    world: ti.template()
):
    is_hit, rec = world.hit(ray, 0.0, ti.math.inf)
    color = vec3(0,0,0)

    if is_hit:
        color = 0.5 * (rec.normal + vec3(1, 1, 1))
    else:
        unit_direction = ray.direction.normalized()
        a = 0.5 * (unit_direction[1] + 1.0)
        color = (1 - a) * vec3(1, 1, 1) + a * vec3(0.5, 0.7, 1.0)

    return color

@ti.kernel
def render(world: ti.template()):
    for i, j in pixels:
        pixel_center = init_pixel_loc + (j * pixel_delta_u) + (i * pixel_delta_v)
        ray_dir = pixel_center - camera_center
        ray = Ray(camera_center, ray_dir)
        pixels[i, j] = ray_color(ray, world)


def main():
    global pixels

    ti.init(arch=ti.gpu)

    pixels = ti.Vector.field(
        n=3,
        dtype=float,
        shape=(
            image_height,
            image_width
        )
    )

    world = HittableList(max_objects=100)
    world.add(Sphere(center=ti.Vector([0.0, 0.0, -1.0]), radius=0.5))
    world.add(Sphere(center=ti.Vector([0.0, -100.5, -1.0]), radius=100.0))

    render(world)
    
    img = pixels.to_numpy()
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save("out.png")


if __name__ == "__main__":
    main()
