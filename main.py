import taichi as ti
from PIL import Image
import numpy as np
import yaml
import os

from typing import TextIO

from utils import *
from hittable import *
from camera import *


def main():
    ti.init(arch=ti.gpu)

    config = None
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    camera = Camera(
        config["camera"]["samples"],
        config["camera"]["image_width"],
        config["camera"]["image_height"],
        config["camera"]["vfov"],
        config["camera"]["max_depth"],
        config["camera"]["gamma"]
    )

    world = HittableList(max_objects=100)
    
    world.add(Sphere(center=ti.Vector([0.0, 0.0, -1.0]), radius=0.5))
    world.add(Sphere(center=ti.Vector([1.0, 1.0, -1.0]), radius=0.25))
    world.add(Sphere(center=ti.Vector([0.0, -5.5, -1.0]), radius=5.0))
    
    live = True

    if live:
        gui = ti.GUI(
            "raytrace test", 
            (camera.image_width, camera.image_height),
            fast_gui=True
        )

        while True:
            if gui.get_event(ti.GUI.ESCAPE):
                break

            camera.handle_motion(gui)
            camera.render(world, camera.kappa)
            gui.set_image(camera.pixels)
            gui.show()

            print(f"  {camera.kappa} | <{camera.position}> | <{camera.rotation}>             ", end="\r")
    else:
        camera.render(world, camera.kappa)
        img = camera.pixels.to_numpy()
        img = img.transpose(1, 0, 2)
        img = np.flip(img, axis=0)
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save("out.png")


if __name__ == "__main__":
    main()
