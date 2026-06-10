import taichi as ti
import taichi.math as tm


vec3 = ti.types.vector(3, float)

@ti.func
def sample_square():
    return vec3(
        ti.random(dtype=float) - 0.5, 
        ti.random(dtype=float) - 0.5, 
        0.0
    )


@ti.func
def random_vector(
    r_min: float = 0.0,
    r_max: float = 1.0
):
    return vec3(
        (ti.random(dtype=float) * (r_max - r_min)) + r_min,
        (ti.random(dtype=float) * (r_max - r_min)) + r_min,
        (ti.random(dtype=float) * (r_max - r_min)) + r_min
    )


@ti.func
def random_on_hemi(normal: vec3):
    on_unit_sphere = random_vector(-1.0, 1.0).normalized()
    result = -on_unit_sphere

    if tm.dot(on_unit_sphere, normal) > 0.0:
        result = on_unit_sphere
    
    return result

@ti.func
def linear_to_gamma(lin_comp):
    result = 0
    
    if lin_comp > 0:
        result =  tm.sqrt(lin_comp)

    return result


@ti.func
def arctanh(x: float) -> float:
    return 0.5 * tm.log((1.0 + x) / (1.0 - x))


def rotate_about_z(vec: vec3, angle: float) -> vec3:
    c, s = tm.cos(angle), tm.sin(angle)
    rot_matrix = tm.mat3([
        [c,  -s,   0.0],
        [s,   c,   0.0],
        [0.0, 0.0, 1.0]
    ])
    
    return rot_matrix @ vec


def rotate_about_y(vec: vec3, angle: float) -> vec3:
    c, s = tm.cos(angle), tm.sin(angle)
    rot_matrix = tm.mat3([
        [c,   0.0, s  ],
        [0.0, 1.0, 0.0],
        [-s,  0.0, c  ]
    ])
    
    return rot_matrix @ vec
