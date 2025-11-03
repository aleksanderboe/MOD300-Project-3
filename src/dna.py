import math
import numpy as np
import matplotlib.pyplot as plt

class SimulationBox:
    """
    A class used to represent a box

    Attributes
    ----------
    x : float
        the upper x bound of the box
    y : float
        the upper y bound of the box
    z : float
        the upper z bound of the box
    """
    def __init__(self, x_upper: float, y_upper: float, z_upper: float, x_lower=0, y_lower=0, z_lower=0):
        self.x_upper = x_upper
        self.y_upper = y_upper
        self.z_upper = z_upper
        self.x_lower = x_lower
        self.y_lower = y_lower
        self.z_lower = z_lower

    def get_volume(self):
        return ((self.x_upper - self.x_lower) *
         (self.y_upper - self.y_lower) *
         (self.z_upper - self.z_lower))

class Point:
    """
    A class used to represent a point

    Attributes
    ----------
    x : float
        the x coordinate of the point
    y : float
        the y coordinate of the point
    z : float
        the z coordinate of the point
    """
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"POINT Center: X {round(self.x, 2)} Y {round(self.y, 2)} Z {round(self.z, 2)}"


class Sphere:
    """
    A class used to represent a sphere

    Attributes
    ----------
    center : Point
        the center of the point
    radius : float
        the radius of the point
    """
    def __init__(self, center: Point, radius: float):
        self.center = center
        self.radius = radius

    def __str__(self):
        return f"SPHERE X: {round(self.center.x, 2)} Y: {round(self.center.y, 2)} Z: {round(self.center.z, 2)} Radius: {round(self.radius, 2)}"

    def is_point_inside(self, point: Point):
        """
        Checks if a point is within 3D Sphere

        :params: A point to check that is within Sphere

        :return: Boolean that represents if point is within sphere
         """

        distance = math.sqrt((point.x - self.center.x)**2 + (point.y - self.center.y)**2 + (point.z - self.center.z)**2)

        return distance <= self.radius
    
    def get_volume(self) -> float:
        """
        calculates volume of 3D Sphere

        :return: float representing volume
        """
        return (4/3) * math.pi * self.radius**3


def create_random_point(box: SimulationBox) -> Point:
    """
     Creates an random point within the 3D simulation box.

     :params: A box

     :return: A Point randomly generated inside the simulation box. 
    """
    x = np.random.uniform(0, box.x_upper)
    y = np.random.uniform(0, box.y_upper)
    z = np.random.uniform(0, box.z_upper)
    return Point(x,y,z)
    

def create_random_sphere(box: SimulationBox) -> Sphere:
    """
    Creates a random sphere within the 3D simulation box.

    :params:
    ----------
    box: SimulationBox
        The box within which to create the sphere.

    :return: Sphere
        A sphere whose entire volume is inside the box.
    """
    max_radius = min(
        box.x_upper - box.x_lower,
        box.y_upper - box.y_lower,
        box.z_upper - box.z_lower
    ) / 2

    radius = np.random.uniform(0, max_radius)

    x = np.random.uniform(box.x_lower + radius, box.x_upper - radius)
    y = np.random.uniform(box.y_lower + radius, box.y_upper - radius)
    z = np.random.uniform(box.z_lower + radius, box.z_upper - radius)

    center = Point(x, y, z)
    return Sphere(center, radius)

def monte_carlo_fraction_inside_sphere(sphere, box, n_points=100_000, plot=False, plot_points=5000):
    """
    Estimates the fraction of points inside a sphere using the Monte Carlo method.

    :params:
    sphere: Sphere
        The sphere to generate points inside
    box: SimulationBox
        The box to generate points inside
    n_points: int
        The number of points to generate
    plot: bool
        Whether to plot the results
    plot_points: int 
        Numbers of random points, used ONLY for plotting. 


    :return: float
        The estimated fraction of points inside the sphere
    """

    if isinstance(sphere, (list, tuple)): 
        spheres = sphere
        estimates = [monte_carlo_fraction_inside_sphere(s, box, n_points=n_points, plot=False) for s in spheres]
        
        if plot:

            points = []
            in_any = []
            for _ in range(plot_points): 
                p = create_random_point(box)
                points.append([p.x, p.y, p.z])
                in_any.append(any(s.is_point_inside(p) for s in spheres))

            points = np.array(points)
            in_any = np.array(in_any)
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            outside = points[~in_any]
            inside = points[in_any]

            ax.scatter(outside[:,0], outside[:,1], outside[:,2], s=4, color='blue', alpha=0.3)
            ax.scatter(inside[:,0], inside[:,1], inside[:,2], s=6, color='red', alpha=0.8)

            for s in spheres:
                ax.scatter(s.center.x, s.center.y, s.center.z, s=40, color='black')

            ax.set_title("Points inside 10 spheres")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            plt.show()

        return estimates
    
    points_inside = 0
    for _ in range(n_points):
        if sphere.is_point_inside(create_random_point(box)): 
            points_inside += 1
    return points_inside / n_points

def estimate_pi(n_points, sphere, box):
    """"
    Estimates π using the Monte Carlo method.

    :params:
    n_points: int
        The number of points to generate
    sphere: Sphere
        The sphere to generate points inside
    box: SimulationBox
        The box to generate points inside

    :return: float
        The estimated value of π
    """
    fraction = monte_carlo_fraction_inside_sphere(sphere, box, n_points, plot=False)
    pi_estimate = (3/4) * fraction * (box.get_volume() / (sphere.radius**3))
    return pi_estimate