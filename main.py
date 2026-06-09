from PIL import Image
import taichi as ti
import numpy as np
import os

from typing import TextIO

from utils import *
from hittable import *
from camera import *

def main():
    ti.init(arch=ti.gpu)
    
    camera = Camera()

    world = HittableList(max_objects=100)
    world.add(Sphere(center=ti.Vector([0.0, 0.0, -1.0]), radius=0.5))
    world.add(Sphere(center=ti.Vector([0.0, -100.5, -1.0]), radius=100.0))

    gui = ti.GUI(
        "raytrace test", 
        (camera.image_width, camera.image_height),
        fast_gui=True
    )

    while True:
        if gui.get_event(ti.GUI.ESCAPE):
            break

        camera.handle_motion(gui)
        camera.render(world)
        gui.set_image(camera.pixels)
        gui.show()

    # camera.render(world)
    # img = camera.pixels.to_numpy()
    # img = img.transpose(1, 0, 2)
    # img = np.flip(img, axis=0)
    # img = (img * 255).astype(np.uint8)
    # Image.fromarray(img).save("out.png")


if __name__ == "__main__":
    main()
