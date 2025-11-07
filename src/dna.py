import math
import numpy as np
import matplotlib.pyplot as plt
import dna 

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

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"X {round(self.x, 2)} Y {round(self.y, 2)} Z {round(self.z, 2)}"


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


    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"X: {round(self.center.x, 2)} Y: {round(self.center.y, 2)} Z: {round(self.center.z, 2)}\nRadius: {round(self.radius, 2)}"

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
    Creates a random point within the 3D simulation box.

    :params:
    box: SimulationBox
        Box defining the bounds for the random point.

    :return: Point
        Randomly generated point inside the simulation box.
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
    Estimate the fraction of random points that fall inside a sphere.

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
    Estimate the fraction of points inside a sphere or list of spheres.

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
    Estimate DNA volume (in nm^3) from atom list using Monte Carlo.

    Returns a tuple: (fraction, dna_volume_nm3, box_volume_nm3, box, spheres)
    """
    spheres = atoms_to_spheres(atoms, units=units)
    box = simulation_box_from_atoms(atoms, units=units, padding_nm=padding_nm)
    fraction = monte_carlo_fraction_inside_union(spheres, box, n_points=n_points, plot=plot)
    dna_volume = fraction * box.get_volume()
    return fraction, dna_volume, box.get_volume(), box, spheres


def read_and_report(filename='dna_coords.txt', n_preview=10):
    """
    Read DNA coordinates and print a short report.

    Returns the list of atom dicts.
    """
    atoms = read_dna_coordinates(filename)
    print(f"Total atoms: {len(atoms)}")
    print("\nFirst {} atoms:".format(min(n_preview, len(atoms))))
    for i, atom in enumerate(atoms[:n_preview]):
        print(f"Atom {i+1}: Element={atom['element']}, X={atom['x']:.2f}, Y={atom['y']:.2f}, Z={atom['z']:.2f}, Radius={atom['radius']:.2f} Å")

    print("\nAtomic radii mapping:")
    for element, radius in ATOMIC_RADII.items():
        print(f"{element}: {radius} Å")

    return atoms


def generate_atoms_box(atoms, units='angstrom', padding_nm=0.5, verbose=True):
    """
    Construct a tight SimulationBox around atoms and return it.

    If verbose=True, prints diagnostics about bounds and volume.
    """
    spheres = atoms_to_spheres(atoms, units=units)
    box = simulation_box_from_atoms(atoms, units=units, padding_nm=padding_nm)
    if verbose:
        x_size = box.x_upper - box.x_lower
        y_size = box.y_upper - box.y_lower
        z_size = box.z_upper - box.z_lower
        print(f"Box bounds (nm):")
        print(f"  x: {box.x_lower:.4f} to {box.x_upper:.4f}  (size {x_size:.4f} nm)")
        print(f"  y: {box.y_lower:.4f} to {box.y_upper:.4f}  (size {y_size:.4f} nm)")
        print(f"  z: {box.z_lower:.4f} to {box.z_upper:.4f}  (size {z_size:.4f} nm)")
        print(f"Box volume: {box.get_volume():.6f} nm^3")
    return box


def estimate_dna_volume_monte_carlo(atoms, sample_sizes=None, units='angstrom', padding_nm=0.5, plot_convergence=True, plot_3d=False):
    """
    Monte Carlo estimates for a list of sample sizes and return results.

    Returns a dict with keys: sample_sizes, fractions, volumes_nm3, box_volume, sum_sphere_vol
    If plot_convergence is True, shows convergence plots. If plot_3d is True, shows a 3D sample plot.
    """
    if sample_sizes is None:
        sample_sizes = [1000, 5000, 10000, 25000, 50000]

    # prepare spheres and box
    spheres = atoms_to_spheres(atoms, units=units)
    box = simulation_box_from_atoms(atoms, units=units, padding_nm=padding_nm)
    box_volume = box.get_volume()

    fractions = []
    volumes = []
    for n in sample_sizes:
        frac = monte_carlo_fraction_inside_union(spheres, box, n_points=n, plot=False)
        v = frac * box_volume
        fractions.append(frac)
        volumes.append(v)

    # plotting
    if plot_convergence:
        plt.figure(figsize=(6,4))
        plt.plot(sample_sizes, fractions, marker='o')
        plt.xscale('log')
        plt.xlabel('Number of samples (log)')
        plt.ylabel('Fraction inside union')
        plt.title('Convergence: fraction inside union')
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(6,4))
        plt.plot(sample_sizes, volumes, marker='o')
        plt.xscale('log')
        plt.xlabel('Number of samples (log)')
        plt.ylabel('Estimated DNA volume (nm^3)')
        plt.title('Convergence: estimated DNA volume')
        plt.grid(True)
        plt.show()

    if plot_3d:
        monte_carlo_fraction_inside_union(spheres, box, n_points=5000, plot=True, plot_points=5000)

    sum_sphere_vol = sum((4/3)*math.pi*(s.radius**3) for s in spheres)

    results = {
        'sample_sizes': sample_sizes,
        'fractions': fractions,
        'volumes_nm3': volumes,
        'box_volume': box_volume,
        'sum_sphere_vol': sum_sphere_vol,
        'box': box,
        'spheres': spheres,
    }
    return results



class Walker: 
    def __init__(self, n_walkers, n_steps): 
        """
        Initalize random walkers 

        n_walkers: int 
            Number of walkers 

        n_steps: int
            Number of steps each walker takes

        """

        self.n_walkers = n_walkers
        self.n_steps = n_steps
        self.start_points = np.random.uniform(-20, 20, size=(n_walkers, 3))

    def random_walkers(self): 
        """
        Generate a set of random walkers in 3D. 
        
        """
        paths = np.zeros((self.n_walkers, self.n_steps + 1, 3))
        paths[:, 0, :] = self.start_points

        for r in range(self.n_walkers): 
            for s in range(1, self.n_steps + 1): 
                step = np.random.choice([-1, 1], size=3)
                paths[r, s] = paths[r, s - 1] + step

        return paths


    def random_walkers_fast(self): 
        """
        Generate fast set of random walkers in 3D. 
        """
        steps = 2*np.random.randint(0, 2, size=(self.n_walkers, self.n_steps, 3)) - 1

        displacement = np.cumsum(steps, axis=1)

        paths = self.start_points[:, None, :] + displacement
        paths = np.concatenate([self.start_points[:, None, :], paths], axis= 1)

        return paths 
    

def plot_walkers(paths, show_start_end=True): 
    """
    Plot 3D random walker paths 
    """

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    n_walkers = paths.shape[0]

    for i in range(n_walkers): 
        ax.plot(paths[i, :, 0], paths [i, :, 1], paths[i, :, 2])

        if show_start_end: 
            ax.scatter(paths[i, 0, 0], paths[i, 0, 1], paths[i, 0, 2], marker='o', s=50, label=f"Start {i+1}")
            ax.scatter(paths[i, -1, 0], paths[i, -1, 1], paths[i, -1, 2], marker='x', s=50, label=f"End {i+1}")

    ax.set_title("3D Random Walk Trajectories")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if show_start_end or n_walkers <= 10:
        _, labels = ax.get_legend_handles_labels()
        if len(labels) > 0: 
            plt.legend()
    plt.show()


def _reflect_to_box(position, box):
    """
    Reflect a 3D point back into the box if it crosses any boundary.

    Parameters
    ----------
    position : np.ndarray
        Point coordinates as a numpy array [x, y, z].
    box : SimulationBox
        Simulation box with lower/upper bounds.

    Returns
    -------
    np.ndarray
        Reflected point inside box bounds.
    """
    x, y, z = position
    if x < box.x_lower:
        x = box.x_lower + (box.x_lower - x)
    if x > box.x_upper:
        x = box.x_upper - (x - box.x_upper)
    if y < box.y_lower:
        y = box.y_lower + (box.y_lower - y)
    if y > box.y_upper:
        y = box.y_upper - (y - box.y_upper)
    if z < box.z_lower:
        z = box.z_lower + (box.z_lower - z)
    if z > box.z_upper:
        z = box.z_upper - (z - box.z_upper)
    return np.array([x, y, z])


def estimate_accessible_volume_walkers(
    atoms,
    probe_radius_nm=0.14,
    n_walkers=200,
    n_steps=2000,
    step_size_nm=0.02,
    units='angstrom',
    padding_nm=0.5,
):
    """
    Estimate accessible volume using random walkers with probe-radius collisions.

    A position is accessible if it lies outside all atomic spheres inflated by the
    probe radius. Walkers move with small isotropic steps; steps that would enter
    an inflated sphere are rejected, and boundary crossings are reflected.

    Parameters
    ----------
    atoms : list
        Atom dicts as returned by `read_dna_coordinates` / `read_and_report`.
    probe_radius_nm : float
        Probe radius in nanometres used to inflate atom spheres.
    n_walkers : int
        Number of independent walkers.
    n_steps : int
        Number of steps per walker.
    step_size_nm : float
        Step size in nanometres for each walker move.
    units : str
        Units of the atom coordinates ('angstrom' or 'nm').
    padding_nm : float
        Padding added to the generated simulation box around atoms.

    Returns
    -------
    tuple
        (accessible_fraction, accessible_volume_nm3, box)
    """
    # Prepare box and inflated spheres
    box = simulation_box_from_atoms(atoms, units=units, padding_nm=padding_nm)
    atom_spheres = atoms_to_spheres(atoms, units=units)
    inflated_spheres = [Sphere(s.center, s.radius + probe_radius_nm) for s in atom_spheres]


    def random_point_in_box(b):
        return np.array([
            np.random.uniform(b.x_lower, b.x_upper),
            np.random.uniform(b.y_lower, b.y_upper),
            np.random.uniform(b.z_lower, b.z_upper),
        ])

    def is_accessible(position):
        p = Point(position[0], position[1], position[2])
        return not any(s.is_point_inside(p) for s in inflated_spheres)

    # Initialize starting positions in accessible space
    starts = []
    for _ in range(n_walkers):
        for _retry in range(10000):
            candidate = random_point_in_box(box)
            if is_accessible(candidate):
                starts.append(candidate)
                break
    if not starts:
        return 0.0, 0.0, box
    starts = np.array(starts)

    accessible_count = 0
    visited_count = 0

    # Run walkers
    for i in range(len(starts)):
        pos = starts[i].copy()
        for _ in range(n_steps):
            direction = np.random.uniform(size=3)
            norm = np.linalg.norm(direction)
            if norm == 0:
                continue
            direction /= norm
            proposal = pos + step_size_nm * direction
            proposal = _reflect_to_box(proposal, box)
            if is_accessible(proposal):
                pos = proposal
                accessible_count += 1
            visited_count += 1

    accessible_fraction = accessible_count / visited_count if visited_count else 0.0
    accessible_volume_nm3 = accessible_fraction * box.get_volume()
    return accessible_fraction, accessible_volume_nm3, box


def validate_accessible_volume_bounds_and_convergence(
    atoms,
    probe_radius_small_nm=0.05,
    probe_radius_large_nm=0.40,
    probe_radius_nm=0.14,
    n_walkers_lo=100,
    n_steps_lo=1000,
    n_walkers_hi=300,
    n_steps_hi=3000,
    step_size_nm=0.02,
    units='angstrom',
    padding_nm=0.5,
):
    """
    Run simple validation checks: bounds, monotonicity (w.r.t. probe radius), and convergence.

    Parameters
    ----------
    atoms : list
        Atom dicts as returned by `read_dna_coordinates`.
    probe_radius_small_nm : float
        Smaller probe radius (nm) for monotonicity check.
    probe_radius_large_nm : float
        Larger probe radius (nm) for monotonicity check.
    probe_radius_nm : float
        Probe radius (nm) for convergence check.
    n_walkers_lo : int
        Fewer walkers for convergence check.
    n_steps_lo : int
        Fewer steps for convergence check.
    n_walkers_hi : int
        More walkers for convergence check.
    n_steps_hi : int
        More steps for convergence check.
    step_size_nm : float
        Step size (nm) for walkers.
    units : str
        Units for atoms ('angstrom' or 'nm').
    padding_nm : float
        Box padding (nm).

    Returns
    -------
    dict
        {
          'box_volume': float,
          'v_small': float,
          'v_large': float,
          'bounds_ok_small': bool,
          'monotonic_ok': bool,
          'v_lo': float,
          'v_hi': float,
          'rel_change': float
        }
    """
    f_small, v_small, box = estimate_accessible_volume_walkers(
        atoms,
        probe_radius_nm=probe_radius_small_nm,
        n_walkers=n_walkers_lo,
        n_steps=n_steps_lo,
        step_size_nm=step_size_nm,
        units=units,
        padding_nm=padding_nm,
    )

    f_large, v_large, _ = estimate_accessible_volume_walkers(
        atoms,
        probe_radius_nm=probe_radius_large_nm,
        n_walkers=n_walkers_lo,
        n_steps=n_steps_lo,
        step_size_nm=step_size_nm,
        units=units,
        padding_nm=padding_nm,
    )

    bounds_ok_small = (0.0 <= v_small <= box.get_volume())
    monotonic_ok = (v_small >= v_large)

    f_lo, v_lo, _ = estimate_accessible_volume_walkers(
        atoms,
        probe_radius_nm=probe_radius_nm,
        n_walkers=n_walkers_lo,
        n_steps=n_steps_lo,
        step_size_nm=step_size_nm,
        units=units,
        padding_nm=padding_nm,
    )
    f_hi, v_hi, _ = estimate_accessible_volume_walkers(
        atoms,
        probe_radius_nm=probe_radius_nm,
        n_walkers=n_walkers_hi,
        n_steps=n_steps_hi,
        step_size_nm=step_size_nm,
        units=units,
        padding_nm=padding_nm,
    )

    rel_change = abs(v_hi - v_lo) / max(v_hi, v_lo) if max(v_hi, v_lo) > 0 else 0.0

    return {
        'box_volume': box.get_volume(),
        'v_small': v_small,
        'v_large': v_large,
        'bounds_ok_small': bounds_ok_small,
        'monotonic_ok': monotonic_ok,
        'v_lo': v_lo,
        'v_hi': v_hi,
        'rel_change': rel_change,
    }