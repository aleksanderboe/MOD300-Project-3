import numpy as np

# NOTE: Bruke dataclass istedenfor 

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
    def __init__(self, x_upper: float, y_upper: float, z_upper: float):
        self.x_upper = x_upper
        self.y_upper = y_upper
        self.z_upper = z_upper

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

def create_random_sphere(box) -> Sphere:
    """
     Creates an random sphere within the 3D simulation box.

     :params: A box

     :return: A Sphere randomly generated inside the simulation box. 
    """
    center = create_random_point(box)
    radius = np.random.uniform(0, box.x_upper) # NOTE: bruke någe annet enn box.x_upper ?

    return Sphere(center, radius)
