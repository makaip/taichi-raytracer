import taichi as ti
import taichi.math as tm
import math

from manifold import Manifold

vec3 = ti.types.vector(3, float)
vec4 = ti.types.vector(4, float)

mat3 = ti.types.matrix(3, 3, float)
mat3x4 = ti.types.matrix(3, 4, float)

STEP_SIZE = 1e-2
HIT_DIST = 1e-2
MAX_DEPTH = 1000


@ti.data_oriented
class Camera:
    pixels: ti.Vector.field

    def __init__(self, manifold: ti.template(), pos: vec3, pitch: float, yaw: float, image_width: int, image_height: int, fov: float):
        self.pos = pos
        self.rot = ti.Vector([0.0, 0.0, 0.0])
        self.speed = 0.1

        self.pitch = pitch
        self.yaw = yaw

        self.image_width = image_width
        self.image_height = image_height
        self.fov = fov

        self.vup = ti.Vector([0.0, 1.0, 0.0])

        th = self.fov * (math.pi / 180)
        h = math.tan(th / 2)

        self.vh = 2 * h
        self.vw = self.vh * (self.image_width / self.image_height)

        self.update(manifold)

        self.pixels = ti.Vector.field(
            n=3,
            dtype=ti.f32,
            shape=(self.image_width, self.image_height)
        )

    @ti.kernel
    def compute_basis(self, manifold: ti.template(), pos: vec3, yaw: float, pitch: float) -> mat3:
        fwd_i = ti.Vector([
            tm.sin(yaw) * tm.cos(pitch),
            tm.sin(pitch),
            -tm.cos(yaw) * tm.cos(pitch)
        ]).normalized()

        up_i = vec3(0.0, 1.0, 0.0)

        basis = manifold.basis(pos)
        g = manifold.metric_tensor(pos, basis)
        g_inv = tm.inverse(g)

        up_prj = tm.dot(fwd_i, g @ fwd_i)

        fwd_l = tm.sqrt(up_prj)
        fwd = fwd_i / fwd_l

        up = up_i - up_prj * fwd
        up_l = tm.sqrt(tm.dot(up, g @ up))
        up = up / up_l

        right_i = g_inv @ tm.cross(fwd, up)
        right_l = tm.sqrt(tm.dot(right_i, g @ right_i))
        right = right_i / right_l

        return ti.Matrix.rows([right, up, fwd])

    def update(self, manifold):
        axes = self.compute_basis(manifold, self.pos, self.yaw, self.pitch)
        
        self.yaw_basis = ti.Vector([axes[0, 0], axes[0, 1], axes[0, 2]])
        self.pitch_basis = ti.Vector([axes[1, 0], axes[1, 1], axes[1, 2]])
        self.rot = ti.Vector([axes[2, 0], axes[2, 1], axes[2, 2]])

        vu = self.vw * self.yaw_basis
        vv = -self.vh * self.pitch_basis

        self.pdu = vu / self.image_width
        self.pdv = vv / self.image_height

        vul = self.pos - self.rot - (vu / 2) - (vv / 2)
        self.p00 = vul + 0.5 * (self.pdu + self.pdv)

    def render(self, manifold, scene):
        self.update(manifold)
        self.render_kernel(manifold, scene, self.pos, self.p00, self.pdu, self.pdv)

    @ti.kernel
    def render_kernel(
        self, 
        manifold: ti.template(),
        scene: ti.template(),

        pos: vec3,
        p00: vec3,
        pdu: vec3,
        pdv: vec3
    ):
        basis = manifold.basis(pos)
        g = manifold.metric_tensor(pos, basis)

        for x, y in self.pixels:
            col_r = 0.0; col_g = 0.0; col_b = 0.0

            rpos = p00 + (x * pdu) + (y * pdv)
            rdir = rpos - pos

            mns = 0.0
            for i in range(3):
                for j in range(3):
                    mns += rdir[i] * g[i,j] * rdir[j]

            scale = 1.0 / tm.sqrt(mns)
            rdir *= scale

            hit, norm, depth = self.march_ray(manifold, scene, rpos, rdir)
            
            depth_col = 1.0
            depth_col = MAX_DEPTH / ((4 * depth) + MAX_DEPTH)

            if hit:
                col_r = tm.clamp((norm.x * 0.5 + 0.5) * depth_col, 0.0, 1.0)
                col_g = tm.clamp((norm.y * 0.5 + 0.5) * depth_col, 0.0, 1.0)
                col_b = tm.clamp((norm.z * 0.5 + 0.5) * depth_col, 0.0, 1.0)

            self.pixels[x, y] = vec3(col_r, col_g, col_b)

    @ti.func
    def march_ray(self, manifold: ti.template(), scene: ti.template(), rpos: vec3, rdir: vec3):
        hit = False
        norm = vec3(0.0)

        tmp_pos = rpos
        tmp_dir = rdir
        steps = 0

        while steps < MAX_DEPTH:
            min_dist = 1e10
            closest_idx = -1

            for idx in range(scene.count[None]):
                dist = scene.objects[idx].sdf(tmp_pos, manifold)

                if (dist < min_dist):
                    min_dist = dist
                    closest_idx = idx
                
            if (min_dist < HIT_DIST):
                basis = manifold.basis(tmp_pos)
                g = manifold.metric_tensor(tmp_pos, basis)
                norm = scene.objects[closest_idx].sdf_normal(tmp_pos, tm.inverse(g), manifold)

                hit = True
                break
            
            dt = tm.min(min_dist * 0.8, 0.4)

            if (dt < 1e-4):
                dt = 1e-4
            
            new_pos, new_dir = manifold.step(tmp_pos, tmp_dir, dt)

            tmp_pos = new_pos
            tmp_dir = new_dir
            steps += 1
        
        return hit, norm, steps

    @ti.kernel
    def step_camera(self, manifold: ti.template(), pos: vec3, vel: vec3) -> vec3:
        new_pos, new_dir = manifold.step(pos, vel, 1.0)
        return new_pos

    def handle_motion(self, gui: ti.GUI, manifold: ti.template()):
        # navigation
        move_dir = ti.Vector([0.0, 0.0, 0.0])
        
        if gui.is_pressed('w'): move_dir[2] += 1
        if gui.is_pressed('s'): move_dir[2] -= 1
        if gui.is_pressed('d'): move_dir[0] += 1
        if gui.is_pressed('a'): move_dir[0] -= 1
        if gui.is_pressed('e'): move_dir[1] += 1
        if gui.is_pressed('q'): move_dir[1] -= 1

        if move_dir.norm() > 0:
            move_dir = move_dir.normalized() * self.speed
            vel = move_dir[0] * self.yaw_basis + move_dir[1] * self.pitch_basis + move_dir[2] * self.rot
            self.pos = self.step_camera(manifold, self.pos, vel)
        
        # rotation
        if gui.is_pressed(ti.GUI.LEFT):
            self.yaw -= 0.05
        if gui.is_pressed(ti.GUI.RIGHT):
            self.yaw += 0.05
        if gui.is_pressed(ti.GUI.UP):
            self.pitch -= 0.05
            self.pitch = max(-math.radians(89), min(math.radians(89), self.pitch))
        if gui.is_pressed(ti.GUI.DOWN):
            self.pitch += 0.05
            self.pitch = max(-math.radians(89), min(math.radians(89), self.pitch))
        
        self.update(manifold)