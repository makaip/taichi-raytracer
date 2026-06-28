import taichi as ti
from PIL import Image
import numpy as np
import yaml
import os
import random

from typing import TextIO

from utils import *
from camera import *
from manifold import *
from shapes import *

def main():
    ti.init(arch=ti.gpu)
    
    config = None
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    camera = Camera(
        ti.Vector([0.0, 0.0, 0.0]),         # pos
        0.0,                                # pitch
        math.radians(180),                  # yaw
        config["camera"]["image_width"],
        config["camera"]["image_height"],
        config["camera"]["fov"]
    )

    manifold = Manifold(h=1e-4)

    scene = Scene(max_objects=100)

    scene.add(Sphere(center=ti.Vector([0.0, 0.0, -1.0]), radius=0.5))
    scene.add(Sphere(center=ti.Vector([1.0, 1.0, -1.0]), radius=0.25))
    scene.add(Sphere(center=ti.Vector([0.0, -5.5, -1.0]), radius=5.0))

    camera.render(manifold, scene)
    
    gui = ti.GUI(
        "Renderer", 
        (camera.image_width, camera.image_height), 
        fast_gui=True
    )

    while gui.running:
        gui.set_image(camera.pixels)
        gui.show()


if __name__ == "__main__":
    main()
