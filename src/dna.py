import math
import numpy as np
import matplotlib.pyplot as plt

# Atomic radii
ATOMIC_RADII = {
    'H': 0.53,  # Hydrogen
    'C': 0.67,  # Carbon
    'N': 0.56,  # Nitrogen
    'O': 0.48,  # Oxygen
    'P': 0.98,  # Phosphorus
}

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
    # sample uniformly within the box bounds (respect lower and upper)
    x = np.random.uniform(box.x_lower, box.x_upper)
    y = np.random.uniform(box.y_lower, box.y_upper)
    z = np.random.uniform(box.z_lower, box.z_upper)
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

def monte_carlo_fraction_inside_sphere(sphere, box, n_points=100_000, plot=False):
    """
    :params:
    sphere: Sphere
        The sphere to generate points inside
    box: SimulationBox
        The box to generate points inside
    n_points: int
        The number of points to generate
    plot: bool
        Whether to plot the results

    :return: float
        The estimated fraction of points inside the sphere
    """
    
    points_inside = 0
    fractions = []

    for i in range(1, n_points + 1):
        point = create_random_point(box)
        if sphere.is_point_inside(point):
            points_inside += 1
        fractions.append(points_inside / i)

    fraction_estimate = points_inside / n_points

    if plot: 

        plt.plot(range(1, n_points + 1), fractions)
        plt.xlabel("Number of points generated")
        plt.ylabel("Fraction inside sphere")
        plt.title("Monte Carlo estimation of points inside sphere")
        plt.grid(True)
        plt.show()
    return fraction_estimate

def fraction_inside_sphere(sphere, box, n_points=100_000, plot=False, plot_points=5000):
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

def read_dna_coordinates(filename):
    """
    Reads a DNA coordinates file and returns a list of dictionaries containing
    atom information (element, coordinates, and radius).

    :params:
    filename: str
        Path to the DNA coordinates file

    :return: list
        List of dictionaries with keys: 'element', 'x', 'y', 'z', 'radius'
    """
    atoms = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if len(parts) >= 5:
                element = parts[1]
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            elif len(parts) == 4:
                element = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            else:
                continue

            element = element.strip().capitalize()

            radius = ATOMIC_RADII.get(element, 0.5)  # default radius if element not found

            atoms.append({
                'element': element,
                'x': x,
                'y': y,
                'z': z,
                'radius': radius
            })

    return atoms


def atoms_to_spheres(atoms, units='angstrom'):
    """
    Convert atom dicts (from read_dna_coordinates) to Sphere objects.

    Parameters
    ----------
    atoms : list
        List of dicts with keys 'element','x','y','z','radius' where positions and radius
        are in the units specified by `units`.
    units : str
        Either 'angstrom' or 'nm'. If 'angstrom', positions and radii are converted to nm.

    Returns
    -------
    list
        List of Sphere objects with positions and radii in nanometres.
    """
    sphs = []
    conv = 1.0
    if units.lower().startswith('ang'):
        conv = 0.1

    for a in atoms:
        x = a['x'] * conv
        y = a['y'] * conv
        z = a['z'] * conv
        r = a['radius'] * conv
        sphs.append(Sphere(Point(x, y, z), r))
    return sphs


def simulation_box_from_atoms(atoms, units='angstrom', padding_nm=0.5):
    """
    Create a SimulationBox that tightly contains all atoms, with optional padding (in nm).

    Parameters
    ----------
    atoms : list
        Atom dicts as returned by `read_dna_coordinates`.
    units : str
        Units of the atom coordinates ('angstrom' or 'nm').
    padding_nm : float
        Extra padding to add to box bounds in nanometres.

    Returns
    -------
    SimulationBox
        Box with lower and upper bounds set to include all atoms and their radii.
    """
    conv = 1.0
    if units.lower().startswith('ang'):
        conv = 0.1

    xs = [a['x'] * conv for a in atoms]
    ys = [a['y'] * conv for a in atoms]
    zs = [a['z'] * conv for a in atoms]
    rs = [a['radius'] * conv for a in atoms]

    x_min = min(x - r for x, r in zip(xs, rs)) - padding_nm
    x_max = max(x + r for x, r in zip(xs, rs)) + padding_nm
    y_min = min(y - r for y, r in zip(ys, rs)) - padding_nm
    y_max = max(y + r for y, r in zip(ys, rs)) + padding_nm
    z_min = min(z - r for z, r in zip(zs, rs)) - padding_nm
    z_max = max(z + r for z, r in zip(zs, rs)) + padding_nm

    return SimulationBox(x_max, y_max, z_max, x_lower=x_min, y_lower=y_min, z_lower=z_min)


def monte_carlo_fraction_inside_union(spheres, box, n_points=100_000, plot=False, plot_points=5000):
    """
    Estimate the fraction of points inside the union of multiple spheres using Monte Carlo.

    Returns fraction in the box (unitless). Use box.get_volume() * fraction to get absolute volume.
    """
    points_inside = 0
    for _ in range(n_points):
        p = create_random_point(box)
        if any(s.is_point_inside(p) for s in spheres):
            points_inside += 1

    fraction = points_inside / n_points

    if plot:
        pts = []
        inside_mask = []
        for _ in range(plot_points):
            p = create_random_point(box)
            pts.append([p.x, p.y, p.z])
            inside_mask.append(any(s.is_point_inside(p) for s in spheres))

        pts = np.array(pts)
        inside_mask = np.array(inside_mask)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        outside = pts[~inside_mask]
        inside = pts[inside_mask]
        if outside.size:
            ax.scatter(outside[:,0], outside[:,1], outside[:,2], s=2, color='blue', alpha=0.25)
        if inside.size:
            ax.scatter(inside[:,0], inside[:,1], inside[:,2], s=4, color='red', alpha=0.6)

        ax.set_title('Points inside union of atomic spheres')
        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Y (nm)')
        ax.set_zlabel('Z (nm)')
        plt.show()

    return fraction


def estimate_dna_volume_from_atoms(atoms, n_points=100_000, units='angstrom', padding_nm=0.5, plot=False):
    """
    High-level helper: estimate DNA volume (in nm^3) from atom list using Monte Carlo.

    Returns a tuple: (fraction, dna_volume_nm3, box_volume_nm3, box, spheres)
    """
    spheres = atoms_to_spheres(atoms, units=units)
    box = simulation_box_from_atoms(atoms, units=units, padding_nm=padding_nm)
    fraction = monte_carlo_fraction_inside_union(spheres, box, n_points=n_points, plot=plot)
    dna_volume = fraction * box.get_volume()
    return fraction, dna_volume, box.get_volume(), box, spheres