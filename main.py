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
from globals import *

@ti.kernel
def populate_scene(manifold: ti.template(), n: int):
    for i in range(n):
        p_r = 5
        p = tm.vec3(
            ti.random() * (2 * p_r) - p_r,
            ti.random() * (2 * p_r) - p_r,
            ti.random() * (2 * p_r) - p_r
        )

        origin_4d = manifold.f(p.x, p.y, p.z)
        
        scene[i] = Sphere(
            origin=origin_4d,
            radius=0.1
        )

def main():
    config = None
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    camera = Camera(
        vec3(0,0,0),
        vec3(0,1,0),
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        config["camera"]["fov"]
    )

    manifold = Manifold(h=1e-4)
    populate_scene(manifold, SCENE_SIZE)
    camera.render(manifold)
    
    gui = ti.GUI(
        "Renderer", 
        (IMAGE_WIDTH, IMAGE_HEIGHT), 
        fast_gui=True
    )

    while gui.running:
        gui.set_image(pixels)
        gui.show()


if __name__ == "__main__":
    main()
