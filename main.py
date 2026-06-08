from PIL import Image
import taichi as ti
import numpy as np
import os

from typing import TextIO

from utils import *
from hittable import *
from camera import *

vec3 = ti.types.vector(3, float)

def main():
    global pixels

    ti.init(arch=ti.gpu)

    pixels = ti.Vector.field(
        n=3,
        dtype=float,
        shape=(
            Camera.image_width,
            Camera.image_height
        )
    )

    world = HittableList(max_objects=100)
    world.add(Sphere(center=ti.Vector([0.0, 0.0, -1.0]), radius=0.5))
    world.add(Sphere(center=ti.Vector([0.0, -100.5, -1.0]), radius=100.0))

    # gui = ti.GUI(
    #     "raytrace test", 
    #     (image_width, image_height),
    #     fast_gui=True
    # )

    # while True:
    #     if gui.get_event(ti.GUI.ESCAPE):
    #         break

    #     render(world)
    #     gui.set_image(pixels)
    #     gui.show()

    Camera.render(world)
    img = pixels.to_numpy()
    img = img.transpose(1, 0, 2)
    img = np.flip(img, axis=0)
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save("out.png")


if __name__ == "__main__":
    main()
