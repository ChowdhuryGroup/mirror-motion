import numpy as np
import matplotlib.pyplot as plt

thetas = [0.0, 0.0]  # horizontal mirror angles
phis = [0.0, 0.0]  # vertical mirror angles


class Vector:
    def __init__(self, components: tuple[float, float, float]):
        self._components = np.array(object=components, dtype=float)
        self._norm = float(np.linalg.norm(self.components))

    @property
    def components(self):
        return self._components

    @property
    def norm(self) -> float:
        return self._norm

    def __truediv__(self, divisor: float):
        new_components = self.components / divisor
        return Vector((new_components[0], new_components[1], new_components[2]))

    def __sub__(self, subtrahend):
        new_components = self.components - subtrahend.components
        return Vector((new_components[0], new_components[1], new_components[2]))

    def dot(self, other):
        return self.components.dot(other.components)


class Ray:
    def __init__(self, source: Vector, direction: Vector):
        self._source = source
        self._direction = direction / direction.norm

    @property
    def source(self):
        return self._source

    @property
    def direction(self):
        return self._direction


class Line:
    def __init__(self, start: Vector, end: Vector):
        self._start = start
        self._end = end

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def length(self):
        if self._length == None:
            self._length = (self.end - self.start).norm
        return self._length


class Plane:
    # Plane defined by Hessian normal form
    def __init__(self, point: np.ndarray, direction: np.ndarray):
        if direction.shape != (3,):
            raise ValueError("Direction needs to be a 3D vector")
        direction_norm = np.linalg.norm(direction)
        if direction_norm == 0:
            raise ValueError("Direction can't have zero length")
        self._normal_vector = direction / direction_norm
        if point.shape != (3,):
            raise ValueError("Point needs to be a 3D point")
        self._origin_distance = -self._normal_vector.dot(point)

    @property
    def normal_vector(self):
        return self._normal_vector


def ray_plane_intersection(ray: Ray, plane: Plane) -> Vector:
    return ray.source + ray.direction * (
        plane._origin_distance - ray.source.dot(plane.normal_vector)
    )


def reflect_ray(ray: Ray, plane: Plane) -> tuple[Line, Ray]:
    intersection = ray_plane_intersection(ray, plane)
    new_direction = (
        ray.direction
        - 2.0 * ray.direction.dot(plane.normal_vector) * plane.normal_vector
    )
    new_ray = Ray(intersection, new_direction)
    old_line_segment = Line(ray.source, intersection)

    return old_line_segment, new_ray


initial_ray = Ray(Vector((0.0, 0.0, 0.0)), Vector((1.0, 0.0, 0.0)))


mirror_spacing = 1.0

# Manual calc
distance_along_second_mirror = (
    mirror_spacing
    * np.sin(2.0 * thetas[0])
    / np.sin(np.pi / 4.0 - 2 * thetas[0] + phis[0])
)
horizontal_displacement = distance_along_second_mirror / np.sin(np.pi / 4.0 + phis[0])


# Vector calc
initial_ray = Ray(Vector((0.0, 0.0, 0.0)), Vector((1.0, 0.0, 0.0)))
first_mirror = Plane(np.array((2.0, 0.0, 0.0)), direction=np.array((-1, 1, 0)))
first_mirror_intersection = initial_ray.intersection(first_mirror)
print(first_mirror_intersection)

mirror_reflect_ray(initial_ray, first_mirror)

# second_mirror_center = first_mirror_center + mirror_spacing * frame.y
# second_mirror = sympy.Plane(second_mirror_center, normal_vector=(-1, -1, 0))
