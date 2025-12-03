import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from mw_plot import MWSkyMap
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
from pathlib import Path

def plt2rgbarr(fig):
    """
    A function to transform a matplotlib figure to a 3d rgb np.array 

    Input
    -----
    fig: matplotlib.figure.Figure
        The plot that we want to encode.        

    Output
    ------
    np.array(ndim, ndim, 3): A 3d map of each pixel in a rgb encoding (the three dimensions are x, y, and rgb)
    
    """
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    rgba_buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h, w, 4))
    return rgba_arr[:, :, :3]

def create_sky_map(center, radius, background):
    mw = MWSkyMap(
        center=center,
        radius=radius,
        background=background,
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    mw.transform(ax)
    plt.title(f"{center} - Radius: {radius[0]} arcsec")
    plt.show()
    return fig, ax


def encode_pixels(img_array): 
    """
    Take the RGB image from plt2rgbarr) and turn it into a 2D array of features for clustering. 

    Encoding: Each pixel is represented by its normalized RGB values. 
    Values are in [0, 1]
    -Blue regions: B is largest
    -Red regions: R is largest
    """
    h, w, _ = img_array.shape

    features = img_array.reshape(-1, 3).astype(np.float32) / 255.0
    return features, (h, w)


def kmeans_cluster_pixels(img_array, n_clusters=4, random_state=0): 
    """
    Parameters
    ----------
    img_array: np.adarray (H, W, 3)
        RGB image from plt2rgbarr.
    n_clusters: int
        Number of color clusters
    random_state: int
        For reproducibility

    Returns
    -------
    label_image: np.adarray (H, W)
        Cluster index for each pixel
    kmeans : sklearn.cluster.KMeans
        Trained KMeans model
    """
    features, (h, w) = encode_pixels(img_array)

    kmeans = KMeans(
        n_clusters=n_clusters, 
        random_state=random_state, 
        n_init="auto", 
    )

    labels = kmeans.fit_predict(features)
    label_image = labels.reshape(h, w)
    return label_image, kmeans


def plot_cluster_labels(label_image, title="K-means pixel clusters"): 
    """
    Visualize the clusters
    """

    plt.figure(figsize=(5,5))
    plt.imshow(label_image, cmap="tab10")
    plt.axis("off")
    plt.title(title)
    plt.show()


def describe_clusters(kmeans): 
    """
    Give a description of each cluster based on its RGB center. 

    Returns a list of strings. 
    """
    descriptions = []
    centers = kmeans.cluster_centers_

    for i, (r, g, b) in enumerate(centers): 
        brightness = (r + g + b) / 3.0

        if brightness < 0.2: 
            desc = "dark background space"
        elif brightness > 0.75: 
            desc = "bright core - stars - white-yellow region"
        elif b == max(r, g, b): 
            desc = "blue-ish region (spiral arms - hot stars)"
        elif r == max(r, g, b): 
            desc = "red-ish region (labels - nebulae )"
        else: 
            desc = "neutral - grey-ish region"
        descriptions.append(
            f"Cluster {i}: {desc} - center RGB (normalized) = [{r: .2f}, {g: .2f}, {b:.2f}]"
        )
    return descriptions

#Task 6
def overlay_cluster_contours(img_array, label_image, title="Original image with overlaid K-means cluster contours"):
    """
    Overlay K-means cluster boundaries as colored contours on top of the original RGB image.

    Parameters
    ----------
    img_array : np.ndarray
        RGB image array of shape (H, W, 3)
    label_image : np.ndarray
        2D array of shape (H, W) with integer cluster labels from kmeans_cluster_pixels.
    title : str, optional
        Title for the plot.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(img_array)
    plt.axis("off")

    n_clusters = int(label_image.max()) + 1
    colors = ["yellow", "cyan", "magenta", "white", "red", "green", "blue"]

    for k in range(n_clusters):
        mask = (label_image == k).astype(float)
        plt.contour(mask, levels=[0.5], colors=colors[k % len(colors)], linewidths=0.8)

    plt.title(title)
    plt.show()

def reconstruct_from_clusters(img_array, label_image, kmeans):
    """
    ----------
    img_array : np.ndarray
        Original RGB image array of shape (H, W, 3)
    label_image : np.ndarray
        2D array of shape (H, W) with integer cluster labels from kmeans_cluster_pixels
    kmeans : sklearn.cluster.KMeans
        Fitted KMeans model with cluster centers
        
    Returns
    -------
    reconstructed : np.ndarray
        Reconstructed image array of shape (H, W, 3) with uint8 dtype
    """
    h, w = label_image.shape
    # Get cluster centers (normalized 0-1)
    centers = kmeans.cluster_centers_
    
    reconstructed = centers[label_image.flatten()].reshape(h, w, 3)
    
    reconstructed = (reconstructed * 255).astype(np.uint8)
    
    return reconstructed

def load_ebola_data(file_path):
    """
    Load Ebola case data from a .dat file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the .dat file containing Ebola data
        
    Returns
    -------
    days : np.ndarray
        Days since first outbreak
    new_cases : np.ndarray
        New cases reported at each time point
    cumulative_cases : np.ndarray
        Cumulative cases over time
    """
    df = pd.read_csv(file_path, sep="\t")
    df["Days"] = pd.to_numeric(df["Days"], errors="coerce")
    df["NumOutbreaks"] = pd.to_numeric(df["NumOutbreaks"], errors="coerce")
    df = df.dropna(subset=["Days", "NumOutbreaks"]).sort_values("Days")
    
    days = df["Days"].values
    new_cases = df["NumOutbreaks"].values
    cumulative_cases = df["NumOutbreaks"].cumsum().values
    
    return days, new_cases, cumulative_cases

def train_linear_regression(days, cumulative_cases):
    """
    Train a linear regression model on cumulative cases vs days.
    
    Parameters
    ----------
    days : np.ndarray
        Days since first outbreak (feature)
    cumulative_cases : np.ndarray
        Cumulative cases (target)
        
    Returns
    -------
    model : sklearn.linear_model.LinearRegression
        Fitted linear regression model
    slope : float
        Slope of the fitted line (cases per day)
    intercept : float
        Intercept of the fitted line
    """
    # Reshape for sklearn (needs 2D array)
    X = days.reshape(-1, 1)
    y = cumulative_cases
    
    # Train linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    slope = model.coef_[0]
    intercept = model.intercept_
    
    return model, slope, intercept

def plot_linear_regression(days, cumulative_cases, model, country_name, ax=None):
    """
    Plot data points and linear regression line.
    
    Parameters
    ----------
    days : np.ndarray
        Days since first outbreak
    cumulative_cases : np.ndarray
        Cumulative cases (data points)
    model : sklearn.linear_model.LinearRegression
        Fitted linear regression model
    country_name : str
        Name of the country
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data points
    ax.scatter(days, cumulative_cases, color='black', marker='o', s=50, 
               label='Data points', zorder=3)
    
    # Generate predictions for smooth line
    days_pred = np.linspace(days.min(), days.max(), 100)
    cumulative_pred = model.predict(days_pred.reshape(-1, 1))
    
    # Plot regression line
    ax.plot(days_pred, cumulative_pred, 'r-', linewidth=2, 
            label='Linear regression', zorder=2)
    
    ax.set_xlabel('Days since first outbreak', fontsize=12)
    ax.set_ylabel('Cumulative Cases', fontsize=12)
    ax.set_title(f'Ebola Epidemic in {country_name}: Linear Regression', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    return ax

def plot_ebola_data(days, new_cases, cumulative_cases, country_name):
    """
    Plot Ebola data in the style of Project 2 Exercise 5 Task 1.
    Shows new cases as red circles and cumulative cases as black line with squares.
    
    Parameters
    ----------
    days : np.ndarray
        Days since first outbreak
    new_cases : np.ndarray
        New cases at each time point
    cumulative_cases : np.ndarray
        Cumulative cases over time
    country_name : str
        Name of the country
    """
    fig, ax_left = plt.subplots()
    ax_right = ax_left.twinx()
    
    # Left y-axis: New cases (red circles)
    ax_left.scatter(
        days, new_cases,
        marker="o",
        facecolors="none",
        edgecolors="red",
        label="New cases"
    )
    ax_left.set_ylabel("Number of outbreaks", color="red")
    ax_left.tick_params(axis='y', labelcolor='red')
    
    # Right y-axis: Cumulative cases (black line with squares)
    ax_right.plot(
        days, cumulative_cases,
        linestyle="-",
        color="black",
        marker="s",
        markerfacecolor="none",
        markeredgecolor="black",
        linewidth=2,
        markersize=5,
        label="Cumulative cases"
    )
    ax_right.set_ylabel("Cumulative number of outbreaks", color="black")
    ax_right.tick_params(axis='y', labelcolor='black')
    
    ax_left.set_xlabel("Days since last outbreak")
    ax_left.set_title(f"Ebola outbreaks in {country_name}")
    ax_left.grid(True, which="both", linestyle=":", alpha=0.6)
    
    plt.show()

def load_all_countries_data(data_dir="MOD300-Project-2/data"):
    """
    Load Ebola data for all three countries.
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing the data files
        
    Returns
    -------
    countries_data : dict
        Dictionary with country names as keys and tuples (days, new_cases, cumulative_cases) as values
    """
    data_path = Path(data_dir)
    
    countries_data = {}
    files = {
        "Guinea": data_path / "ebola_cases_guinea.dat",
        "Liberia": data_path / "ebola_cases_liberia.dat",
        "Sierra Leone": data_path / "ebola_cases_sierra_leone.dat",
    }
    
    for country, file_path in files.items():
        if file_path.exists():
            days, new_cases, cumulative_cases = load_ebola_data(file_path)
            countries_data[country] = (days, new_cases, cumulative_cases)
        else:
            print(f"Warning: File not found: {file_path}")
    
    return countries_data
#Task 7
def run_task7_experiments(img_array, k_values=(3, 5)): 
    """
    Try different numbers of clusters, repeat clustering and overlay for each k, and then print cluster descriptions. 

    Parameters: 
    ----------
    img_array: np.adarray
        RGB image array from plt2rgbarr
    k_values: int 
        Different number of clusters to test. 
    """
    for k in k_values: 
        label_image, kmeans = kmeans_cluster_pixels(img_array, n_clusters=k)
        
        plot_cluster_labels(label_image, title=f"K-means Clusters (k={k})")

        for desc in describe_clusters(kmeans): 
            print(desc)
        
        overlay_cluster_contours(img_array, label_image, title=f"Original image with overlaid K-means contours (k={k})")

def load_ebola_data(file_path):
    """
    Load Ebola case data from a .dat file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the .dat file containing Ebola data
        
    Returns
    -------
    days : np.ndarray
        Days since first outbreak
    new_cases : np.ndarray
        New cases reported at each time point
    cumulative_cases : np.ndarray
        Cumulative cases over time
    """
    df = pd.read_csv(file_path, sep="\t")
    df["Days"] = pd.to_numeric(df["Days"], errors="coerce")
    df["NumOutbreaks"] = pd.to_numeric(df["NumOutbreaks"], errors="coerce")
    df = df.dropna(subset=["Days", "NumOutbreaks"]).sort_values("Days")
    
    days = df["Days"].values
    new_cases = df["NumOutbreaks"].values
    cumulative_cases = df["NumOutbreaks"].cumsum().values
    
    return days, new_cases, cumulative_cases

def train_linear_regression(days, cumulative_cases):
    """
    Train a linear regression model on cumulative cases vs days.
    
    Parameters
    ----------
    days : np.ndarray
        Days since first outbreak (feature)
    cumulative_cases : np.ndarray
        Cumulative cases (target)
        
    Returns
    -------
    model : sklearn.linear_model.LinearRegression
        Fitted linear regression model
    slope : float
        Slope of the fitted line (cases per day)
    intercept : float
        Intercept of the fitted line
    """
    # Reshape for sklearn (needs 2D array)
    X = days.reshape(-1, 1)
    y = cumulative_cases
    
    # Train linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    slope = model.coef_[0]
    intercept = model.intercept_
    
    return model, slope, intercept

def plot_linear_regression(days, cumulative_cases, model, country_name, ax=None):
    """
    Plot data points and linear regression line.
    
    Parameters
    ----------
    days : np.ndarray
        Days since first outbreak
    cumulative_cases : np.ndarray
        Cumulative cases (data points)
    model : sklearn.linear_model.LinearRegression
        Fitted linear regression model
    country_name : str
        Name of the country
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data points
    ax.scatter(days, cumulative_cases, color='black', marker='o', s=50, 
               label='Data points', zorder=3)
    
    # Generate predictions for smooth line
    days_pred = np.linspace(days.min(), days.max(), 100)
    cumulative_pred = model.predict(days_pred.reshape(-1, 1))
    
    # Plot regression line
    ax.plot(days_pred, cumulative_pred, 'r-', linewidth=2, 
            label='Linear regression', zorder=2)
    
    ax.set_xlabel('Days since first outbreak', fontsize=12)
    ax.set_ylabel('Cumulative Cases', fontsize=12)
    ax.set_title(f'Ebola Epidemic in {country_name}: Linear Regression', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    return ax

def plot_ebola_data(days, new_cases, cumulative_cases, country_name):
    """
    Plot Ebola data in the style of Project 2 Exercise 5 Task 1.
    Shows new cases as red circles and cumulative cases as black line with squares.
    
    Parameters
    ----------
    days : np.ndarray
        Days since first outbreak
    new_cases : np.ndarray
        New cases at each time point
    cumulative_cases : np.ndarray
        Cumulative cases over time
    country_name : str
        Name of the country
    """
    fig, ax_left = plt.subplots()
    ax_right = ax_left.twinx()
    
    # Left y-axis: New cases (red circles)
    ax_left.scatter(
        days, new_cases,
        marker="o",
        facecolors="none",
        edgecolors="red",
        label="New cases"
    )
    ax_left.set_ylabel("Number of outbreaks", color="red")
    ax_left.tick_params(axis='y', labelcolor='red')
    
    # Right y-axis: Cumulative cases (black line with squares)
    ax_right.plot(
        days, cumulative_cases,
        linestyle="-",
        color="black",
        marker="s",
        markerfacecolor="none",
        markeredgecolor="black",
        linewidth=2,
        markersize=5,
        label="Cumulative cases"
    )
    ax_right.set_ylabel("Cumulative number of outbreaks", color="black")
    ax_right.tick_params(axis='y', labelcolor='black')
    
    ax_left.set_xlabel("Days since last outbreak")
    ax_left.set_title(f"Ebola outbreaks in {country_name}")
    ax_left.grid(True, which="both", linestyle=":", alpha=0.6)
    
    plt.show()

def load_all_countries_data(data_dir="MOD300-Project-2/data"):
    """
    Load Ebola data for all three countries.
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing the data files
        
    Returns
    -------
    countries_data : dict
        Dictionary with country names as keys and tuples (days, new_cases, cumulative_cases) as values
    """
    data_path = Path(data_dir)
    
    countries_data = {}
    files = {
        "Guinea": data_path / "ebola_cases_guinea.dat",
        "Liberia": data_path / "ebola_cases_liberia.dat",
        "Sierra Leone": data_path / "ebola_cases_sierra_leone.dat",
    }
    
    for country, file_path in files.items():
        if file_path.exists():
            days, new_cases, cumulative_cases = load_ebola_data(file_path)
            countries_data[country] = (days, new_cases, cumulative_cases)
        else:
            print(f"Warning: File not found: {file_path}")
    
    return countries_data