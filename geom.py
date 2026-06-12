import taichi as ti
import taichi.math as tm

import numpy as np

from utils import arctanh

vec3 = ti.types.vector(3, float)

# https://geoopt.readthedocs.io/en/latest/extended/stereographic.html
# https://andbloch.github.io/K-Stereographic-Model/

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
    denom = 1.0 - 2.0 * k * xy + k * k * x2 * y2
    
    return num / (tm.sign(denom) * tm.max(abs(denom), 1e-15))


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
    
    result = p

    if v_norm >= 1e-7:
        if ti.abs(k) < 1e-6:
            result = mobius_add(p, v, k)
        else:
            sqrt_k = ti.sqrt(ti.abs(k))

            pp = p.dot(p)
            lam = 1.0 / (1.0 + k * pp)

            t = 0
            if k < 0.0:
                t = ti.tanh(sqrt_k * lam * v_norm * 0.5) / sqrt_k
            else:
                t = ti.tan(sqrt_k * lam * v_norm * 0.5) / sqrt_k
            
            y = mobius_add(p, t * v / v_norm, k)
        
            result = y

    return result


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
    result = vec3(0.0)

    if d >= 1e-7:    
        if abs(k) < 1e-6:
            result = diff
        else:
            sqrt_k = ti.sqrt(ti.abs(k))

            pp = p.dot(p)
            lam = 1.0 / (1.0 + k * pp)

            s = 0
            if k < 0.0:
                # custom taichi atanh implementation bc tm doesent have it
                s = 2.0 * (lam / sqrt_k) * arctanh(sqrt_k * d)
            else:
                s = 2.0 * (lam / sqrt_k) * tm.atan2(sqrt_k * d, 1.0)
            
            result = s * diff / d
    
    return result


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



@ti.func
def rk4_step(p: vec3, v: vec3, h: float, k: float):
    """
    https://www.geeksforgeeks.org/dsa/runge-kutta-4th-order-method-solve-differential-equation/
    https://lpsa.swarthmore.edu/NumInt/NumIntFourth.html
    """
    
    dp1 = v;                dv1 = geodesic_accel(p, v, k)       # k1 derivative at given point

    p2  = p + 0.5*h*dp1;     v2 = v + 0.5*h*dv1                 # midpoint from k1
    dp2 = v2;               dv2 = geodesic_accel(p2, v2, k)     # k2 derivative at midpoint
 
    p3  = p + 0.5*h*dp2;     v3 = v + 0.5*h*dv2                 # midpoint from k2
    dp3 = v3;               dv3 = geodesic_accel(p3, v3, k)     # k3 derivative at midpoint

    p4  = p + h*dp3;         v4 = v + h*dv3
    dp4 = v4;               dv4 = geodesic_accel(p4, v4, k)     # endpoint estimate from k3

    p_new = p + (h/6.0) * (dp1 + 2.0*dp2 + 2.0*dp3 + dp4)       # estimate of position(t_0 + h)
    v_new = v + (h/6.0) * (dv1 + 2.0*dv2 + 2.0*dv3 + dv4)       # estimate of velocity(t_0 + h)

    # preserve speed
    v_new = v_new.normalized() * v.norm()

    # handle hyperbolic geom bc coords become singular at r = 1 / sqrt(-k)
    if k < -1e-6:
        r_max = 0.99 / ti.sqrt(-k)
        pn = p_new.norm()

        if pn > r_max:
            p_new = p_new * (r_max / pn)

    return p_new, v_new
