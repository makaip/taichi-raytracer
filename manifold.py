import taichi as ti
import taichi.math as tm

from utils import *


vec3 = ti.types.vector(3, float)
vec4 = ti.types.vector(4, float)

mat3 = ti.types.matrix(3, 3, float)
mat3x4 = ti.types.matrix(3, 4, float)

@ti.data_oriented
class Manifold:
    h: float

    def __init__(self, h: float):
        self.h = h

    @ti.func
    def f(self, u: float, v: float, w: float) -> vec4:
        return vec4(
            u,
            v,
            w,
            1
        )

        # return vec4(
        #     tm.sqrt(1.0 + u**2 + v**2 + w**2),
        #     u,
        #     v,
        #     w
        # )

        # return vec4(
        #     ti_cosh(u),
        #     ti_sinh(u) * ti.cos(v),
        #     ti_sinh(u) * ti.sin(v) * ti.cos(w),
        #     ti_sinh(u) * ti.sin(v) * ti.sin(w),
        # )

        # return vec4(
        #     ti.cos(u),
        #     ti.sin(u) * ti.cos(v),
        #     ti.sin(u) * ti.sin(v) * ti.cos(w),
        #     ti.sin(u) * ti.sin(v) * ti.sin(w),
        # )

    @ti.func
    def basis(self, pos: vec3) -> mat3x4:
        r_u = (self.f(pos.x + self.h, pos.y, pos.z) - self.f(pos.x - self.h, pos.y, pos.z)) / (2 * self.h)
        r_v = (self.f(pos.x, pos.y + self.h, pos.z) - self.f(pos.x, pos.y - self.h, pos.z)) / (2 * self.h)
        r_w = (self.f(pos.x, pos.y, pos.z + self.h) - self.f(pos.x, pos.y, pos.z - self.h)) / (2 * self.h)

        return mat3x4([
            [r_u[0], r_u[1], r_u[2], r_u[3]],
            [r_v[0], r_v[1], r_v[2], r_v[3]],
            [r_w[0], r_w[1], r_w[2], r_w[3]]
        ])
    
    @ti.func
    def metric_tensor(self, pos: vec3, basis: mat3x4) -> mat3:
        g = mat3(0.0)
        for i in range(3):
            for k in range(3):
                for l in range(4):
                    g[i, k] += basis[i, l] * basis[k, l]
        return g
        
    @ti.func
    def proj_manifold(self, pos: vec3, target: vec4, iters: int = 10) -> vec3:
        for i in range(iters):
            # jacobean "J" = matrix of bases
            J = self.basis(pos)
            g = self.metric_tensor(pos, J)
            g_inv = tm.inverse(g)

            cur = self.f(pos.x, pos.y, pos.z)
            res = cur - target
            
            if (tm.dot(res, res) < 1e-2):
                break
            
            Jtr = vec3(0.0)

            # dot prod
            for j in range(3):
                for l in range(4):
                    Jtr[j] += J[j, l] * res[l]

            delta = g_inv @ (-Jtr)
            pos += delta
        
        return pos

    @ti.func
    def step(self, pos: vec3, vel: vec3, dt: float):
        J1 = self.basis(pos)
        ru = J1[0,:]; rv = J1[1,:]; rw = J1[2,:]

        pos4 = self.f(pos.x, pos.y, pos.z)
        vel4 = ru * vel.x + rv * vel.y + rw * vel.z

        new_pos4 = pos4 + vel4 * dt
        new_pos = self.proj_manifold(pos, new_pos4)

        J2 = self.basis(new_pos)
        g = self.metric_tensor(new_pos, J2)
        g_inv = tm.inverse(g)

        new_vel = vec3(0.0)
        for i in range(3):
            for j in range(3):
                new_vel[i] += g_inv[i,j] * tm.dot(J2[j,:], vel4)
        
        return [new_pos, new_vel]

