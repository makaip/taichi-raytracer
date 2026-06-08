from hittable import *

vec3 = ti.types.vector(3, float)

@ti.data_oriented
class Camera:
    pixels: ti.Vector.field
    def __init__(self):
        self.image_width = 800
        self.image_height = 450

        self.focal_length = 1.0
        self.view_height = 2.0
        self.view_width = self.view_height * (self.image_width / self.image_height)

        # use ti.Vector outside @ti scope
        self.camera_center = ti.Vector([0.0, 0.0, 0.0])

        self.view_u = ti.Vector([self.view_width, 0.0, 0.0])
        self.view_v = ti.Vector([0.0, self.view_height, 0.0])

        self.pixel_delta_u = self.view_u / self.image_width
        self.pixel_delta_v = self.view_v / self.image_height

        self.view_upper_left = self.camera_center - \
            ti.Vector([0.0, 0.0, self.focal_length]) - \
            (self.view_u / 2.0) - (self.view_v / 2.0)
        self.init_pixel_loc = self.view_upper_left + \
            0.5 * (self.pixel_delta_u + self.pixel_delta_v)
            
        self.pixels = ti.Vector.field(
            n=3,
            dtype=float,
            shape=(
                self.image_width,
                self.image_height
            )
        )

    @ti.func
    def ray_color(
            self,
            ray: Ray,
            world: ti.template()
    ) -> vec3:
        is_hit, rec = world.hit(ray, Interval(0, ti.math.inf))
        color = vec3(0, 0, 0)

        if is_hit:
            color = 0.5 * (rec.normal + vec3(1, 1, 1))
        else:
            unit_direction = ray.direction.normalized()
            a = 0.5 * (unit_direction[1] + 1.0)
            color = (1 - a) * vec3(1, 1, 1) + a * vec3(0.5, 0.7, 1.0)

        return color

    @ti.kernel
    def render(
        self, 
        world: ti.template()
    ):
        for i, j in self.pixels:
            pixel_center = self.init_pixel_loc + (i * self.pixel_delta_u) + (j * self.pixel_delta_v)
            ray_dir = pixel_center - self.camera_center
            ray = Ray(self.camera_center, ray_dir)
            self.pixels[i, j] = self.ray_color(ray, world)
