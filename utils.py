import taichi as ti
import taichi.math as tm


vec3 = ti.types.vector(3, float)



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
