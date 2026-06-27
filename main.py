import taichi as ti
from PIL import Image
import numpy as np
import yaml
import os

from typing import TextIO

from utils import *
from camera import *

def main():
    ti.init(arch=ti.gpu)

    config = None
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    


if __name__ == "__main__":
    main()
