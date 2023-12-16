import numpy as np
import matplotlib.pyplot as plt

thetas = [0.0, 0.0]  # horizontal mirror angles
phis = [0.0, 0.0]  # vertical mirror angles


class Ray:
    def __init__(self, source: np.ndarray, direction: np.ndarray):
        self._source = source
        if direction.shape != (3,):
            raise ValueError("Source needs to be a 3D point")
        self._direction = direction / np.linalg.norm(direction)

    @property
    def source(self):
        return self._source

    @property
    def direction(self):
        return self._direction


class Line:
    def __init__(self, start: np.ndarray, end: np.ndarray):
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
            self._length = np.linalg.norm(self.end - self.start)
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


def ray_plane_intersection(ray: Ray, plane: Plane):
    return ray.source


initial_ray = Ray(np.array((0.0, 0.0, 0.0)), np.array((1.0, 0.0, 0.0)))


mirror_spacing = 1.0

# Manual calc
distance_along_second_mirror = (
    mirror_spacing
    * sympy.sin(2.0 * thetas[0])
    / sympy.sin(sympy.pi / 4.0 - 2 * thetas[0] + phis[0])
)
horizontal_displacement = distance_along_second_mirror / sympy.sin(
    sympy.pi / 4.0 + phis[0]
)


def mirror_reflect_ray(ray: sympy.Ray3D, mirror: sympy.Plane):
    initial_direction = ray.direction_cosine
    print(mirror.equation().as_coefficients_dict())
    normal = sympy.vector.Vector(mirror.normal)
    input("wtf")
    normalized_normal = normal.normalize()
    input("brah")
    exit_direction = (
        initial_direction
        - 2 * (initial_direction.dot(normalized_normal)) * normalized_normal
    )
    intersection_point = ray.intersection(mirror)
    return sympy.Ray3D(intersection_point, exit_direction)


# Sympy vector calc
first_mirror_center = sympy.Point(1.0, 0.0, 0.0)
first_mirror = sympy.Plane(first_mirror_center, normal_vector=(-1, 1, 0))
first_mirror_intersection = initial_ray.intersection(first_mirror)
print(first_mirror_intersection)

mirror_reflect_ray(initial_ray, first_mirror)

# second_mirror_center = first_mirror_center + mirror_spacing * frame.y
# second_mirror = sympy.Plane(second_mirror_center, normal_vector=(-1, -1, 0))
