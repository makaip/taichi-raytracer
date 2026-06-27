import taichi as ti
import taichi.math as tm
import yaml

config = None
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

IMAGE_WIDTH = config["camera"]["image_width"]
IMAGE_HEIGHT = config["camera"]["image_height"]

pixels = ti.Field(IMAGE_WIDTH, IMAGE_HEIGHT)
