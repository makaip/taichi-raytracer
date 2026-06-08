from hittable import *

pixels = None

vec3 = ti.types.vector(3, float)

@ti.dataclass
class Camera:
    image_height: int
    center: vec3
    pixel00_loc: vec3
    pixel_delta_u: vec3
    pixel_delta_v: vec3

    def __init__(self):
        image_width = 800
        image_height = 450

        pixels = ti.Vector.field(
            n=3,
            dtype=float,
            shape=(
                image_width,
                image_height
            )
        )

        focal_length = 1.0
        view_height = 2.0
        view_width = view_height * (image_width / image_height)

        # use ti.Vector outside @ti scope
        camera_center = ti.Vector([0.0, 0.0, 0.0])

        view_u = ti.Vector([view_width, 0.0, 0.0])
        view_v = ti.Vector([0.0, view_height, 0.0])

        pixel_delta_u = view_u / image_width
        pixel_delta_v = view_v / image_height

        view_upper_left = camera_center - \
            ti.Vector([0.0, 0.0, focal_length]) - \
            (view_u / 2.0) - (view_v / 2.0)
        init_pixel_loc = view_upper_left + \
            0.5 * (pixel_delta_u + pixel_delta_v)

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
    def render(self, world: ti.template()):
        for i, j in pixels:
            pixel_center = self.init_pixel_loc + (i * self.pixel_delta_u) + (j * self.pixel_delta_v)
            ray_dir = pixel_center - self.camera_center
            ray = Ray(self.camera_center, ray_dir)
            pixels[i, j] = self.ray_color(ray, world)

