import taichi as ti
import taichi.math as tm

import numpy as np

from utils import arctanh

vec3 = ti.types.vector(3, float)

# https://geoopt.readthedocs.io/en/latest/extended/stereographic.html


@ti.func
def mobius_add(
    x: vec3,    # first point
    y: vec3,    # second point
    k: float    # curvature
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


@ti.func
def exp_map(
    p: vec3,    # origin point
    v: vec3,    # direction to shoot
    k: float    # curvature
) -> vec3:
    """
    exponential map
    https://geoopt.readthedocs.io/en/latest/_modules/geoopt/manifolds/stereographic/math.html#expmap
    https://geoopt.readthedocs.io/en/latest/_modules/geoopt/manifolds/stereographic/math.html#tan_k
    https://math.stackexchange.com/questions/3766220/what-is-exponential-map-in-differential-geometry
    """

    v_norm = v.norm()

    if v_norm < 1e-7:
        return p
    if ti.abs(k) < 1e-6:
        return p + v
    
    sqrt_k = ti.sqrt(ti.abs(k))

    if k < 0.0:
        t = ti.tanh(sqrt_k * v_norm * 0.5) / sqrt_k
    else:
        t = ti.tan(sqrt_k * v_norm * 0.5) / sqrt_k
    
    y = mobius_add(p, t * v / v_norm, k)
    return y


@ti.func
def log_map(
    p: vec3,    # origin point
    q: vec3,    # direction to point
    k: float    # curvature
) -> vec3:
    """
    tangent vector at p pointing toward q
    https://geoopt.readthedocs.io/en/latest/_modules/geoopt/manifolds/stereographic/math.html#logmap
    https://geoopt.readthedocs.io/en/latest/_modules/geoopt/manifolds/stereographic/math.html#lambda_x
    """

    minus_p = -p
    diff = mobius_add(minus_p, q, k)
    d = diff.norm()

    if d < 1e-7:
        return vec3(0.0)
    
    sqrt_k = ti.sqrt(ti.abs(k))

    if k < 0.0:
        # custom taichi atanh implementation bc tm doesent have it
        s = (2.0 / sqrt_k) * arctanh(sqrt_k * d)
    else:
        s = (2.0 / sqrt_k) * tm.atan2(sqrt_k * d, 1.0)
    
    return s * diff / d


@ti.func
def geodesic_dist(
    p: vec3,    # point on manifold
    q: vec3,    # point on manifold
    k: float    # curvature
) -> float:
    """
    returns geodesic distance
    https://geoopt.readthedocs.io/en/latest/_modules/geoopt/manifolds/stereographic/math.html#dist
    """

    return log_map(p, q, k).norm()


@ti.func
def geodesic_accel(
    p: vec3,    # point
    v: vec3,    # velocity
    k: float    # curvature
) -> vec3:
    """
    closed-form geodesic acceleration for the k-stereographic model
    dv / dt = -2(kappa) * (<p, v> v - <v, v> p) / (conformal factor)^2
    where conformal factor (lambda) = 1 / (1 + k (||p||)^2)
    https://geoopt.readthedocs.io/en/latest/_modules/geoopt/manifolds/stereographic/math.html#lambda_x
    https://geoopt.readthedocs.io/en/latest/_modules/geoopt/manifolds/stereographic/math.html#inner
    """

    pp = p.dot(p)
    pv = p.dot(v)
    vv = v.dot(v)
    lam = 1.0 / (1.0 + k * pp)
    return -2.0 * k * lam * lam * (pv * v - vv * p)

