from PIL import Image
import taichi as ti
import numpy as np
import os

from typing import TextIO

from ray import Ray

vec3 = ti.types.vector(3, float)

image_width, image_height = 400, 225
pixels = None

focal_length = 1.0

view_height = 2
view_width = view_height * (image_width / image_height)
camera_center = vec3(0, 0, 0)

view_u = vec3(view_width, 0, 0)
view_v = vec3(0, -view_height, 0)

pixel_delta_u = view_u / image_width
pixel_delta_v = view_v / image_height

view_upper_left = camera_center - vec3(0, 0, focal_length) - view_u / 2 - view_v / 2
init_pixel_loc = view_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v)

@ti.func
def hit_sphere(center: vec3, radius: float, ray: Ray):
    oc = center - ray.origin

    a = ti.math.dot(ray.direction, ray.direction)
    b = -2 * ti.math.dot(ray.direction, oc)
    c = ti.math.dot(oc, oc) - (radius ** 2)

    disc = (b ** 2) - (4 * a * c)

    t = -1.0

    if disc >= 0:
        t = (-b - ti.sqrt(disc)) / (2.0 * a)
    
    return t

@ti.func
def ray_color(ray: Ray):
    t = hit_sphere(vec3(0, 0, -1), 0.5, ray)
    color = vec3(0, 0, 0)

    if t > 0:
        normal = (ray.at(t) - vec3(0, 0, -1)).normalized()
        color = 0.5 * vec3(normal.x + 1, normal.y + 1, normal.z + 1)
    else:
        unit_direction = ray.direction.normalized()
        a = 0.5 * (unit_direction[1] + 1)
        color = (1 - a) * vec3(1, 1, 1) + a * vec3(0.5, 0.7, 1.0)

    return color

@ti.kernel
def render():
    for i, j in pixels:
        pixel_center = init_pixel_loc + (j * pixel_delta_u) + (i * pixel_delta_v)
        ray_dir = pixel_center - camera_center
        ray = Ray(camera_center, ray_dir)
        pixels[i, j] = ray_color(ray)


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

    render()

    img = pixels.to_numpy()
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save("out.png")


if __name__ == "__main__":
    main()
