import taichi as ti
import taichi.math as tm

import numpy as np

vec3 = ti.types.vector(3, float)

# https://geoopt.readthedocs.io/en/latest/extended/stereographic.html


@ti.func
def mobius_add(
    x: vec3,
    y: vec3,
    k: float
) -> vec3:
    """
    mobius addition. lowk just yoinked from this ML library:
    https://geoopt.readthedocs.io/en/latest/_modules/geoopt/manifolds/stereographic/math.html#mobius_add
    """

    x2 = x.dot(x)
    y2 = y.dot(y)
    xy = x.dot(y)

    num = (1.0 - 2.0 * k * xy - k * y2) * x + (1.0 + k * x2) * y
    denom = 1.0 - 2.0 * k * xy + k *k * x2 * y2
    
    return num / tm.clamp(denom, xmax=1e-15)


# https://proceedings.neurips.cc/paper_files/paper/2018/file/dbab2adc8f9d078009ee3fa810bea142-Paper.pdf
# https://math.stackexchange.com/questions/3766220/what-is-exponential-map-in-differential-geometry
# https://www.math.uni-hamburg.de/home/lindemann/material/DG2020L17_slides.pdf
# http://staff.ustc.edu.cn/~wangzuoq/Courses/16S-RiemGeom/Notes/Lec13.pdf
