import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from mw_plot import MWSkyMap
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from ebola_utils import (
    load_ebola_data,
    train_linear_regression,
    plot_linear_regression,
    plot_ebola_data,
    load_all_countries_data
)

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
    centers = kmeans.cluster_centers_
    
    reconstructed = centers[label_image.flatten()].reshape(h, w, 3)
    
    reconstructed = (reconstructed * 255).astype(np.uint8)
    
    return reconstructed

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

# Task 2: Better fitting functions for Ebola data
def train_polynomial_regression(days, cumulative_cases, degree=2):
    """
    Train a polynomial regression model on cumulative cases vs days.
    
    Parameters
    ----------
    days : np.ndarray
        Days since first outbreak (feature)
    cumulative_cases : np.ndarray
        Cumulative cases (target)
    degree : int, default=2
        Degree of the polynomial (2 for quadratic, 3 for cubic, etc.)
        
    Returns
    -------
    model : sklearn.pipeline.Pipeline
        Fitted polynomial regression model
    r2_score : float
        R² score of the model
    """
    X = days.reshape(-1, 1)
    y = cumulative_cases
    
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    model.fit(X, y)
    
    y_pred = model.predict(X)
    score = r2_score(y, y_pred)
    
    return model, score

def plot_polynomial_regression(days, cumulative_cases, model, country_name, ax=None):
    """
    Plot data points and polynomial regression curve.
    
    Parameters
    ----------
    days : np.ndarray
        Days since first outbreak
    cumulative_cases : np.ndarray
        Cumulative cases (data points)
    model : sklearn.pipeline.Pipeline
        Fitted polynomial regression model
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
    
    ax.scatter(days, cumulative_cases, color='black', marker='o', s=50, 
               label='Data points', zorder=3)
    
    days_pred = np.linspace(days.min(), days.max(), 100)
    cumulative_pred = model.predict(days_pred.reshape(-1, 1))
    
    ax.plot(days_pred, cumulative_pred, 'b-', linewidth=2, 
            label='Polynomial regression', zorder=2)
    
    ax.set_xlabel('Days since first outbreak', fontsize=12)
    ax.set_ylabel('Cumulative Cases', fontsize=12)
    ax.set_title(f'Ebola Epidemic in {country_name}: Polynomial Regression', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    return ax

def compare_models(days, cumulative_cases, country_name):
    """
    Compare linear and polynomial regression models for Ebola data.
    
    Parameters
    ----------
    days : np.ndarray
        Days since first outbreak
    cumulative_cases : np.ndarray
        Cumulative cases (target)
    country_name : str
        Name of the country
        
    Returns
    -------
    results : dict
        Dictionary containing models and their R² scores
    """
    results = {}
    
    # Linear regression
    linear_model, _, _ = train_linear_regression(days, cumulative_cases)
    linear_pred = linear_model.predict(days.reshape(-1, 1))
    linear_r2 = r2_score(cumulative_cases, linear_pred)
    results['linear'] = {'model': linear_model, 'r2': linear_r2, 'name': 'Linear'}
    
    poly2_model, poly2_r2 = train_polynomial_regression(days, cumulative_cases, degree=2)
    results['polynomial_2'] = {'model': poly2_model, 'r2': poly2_r2, 'name': 'Polynomial (degree 2)'}
    
    poly3_model, poly3_r2 = train_polynomial_regression(days, cumulative_cases, degree=3)
    results['polynomial_3'] = {'model': poly3_model, 'r2': poly3_r2, 'name': 'Polynomial (degree 3)'}
    
    print(f"\nModel Comparison for {country_name}:")
    print("-" * 50)
    for key, value in results.items():
        print(f"{value['name']:25s} R² = {value['r2']:.6f}")
    
    best_key = max(results.keys(), key=lambda k: results[k]['r2'])
    best_model = results[best_key]
    print(f"\nBest model: {best_model['name']} (R² = {best_model['r2']:.6f})")
    
    return results

def plot_model_comparison(days, cumulative_cases, models_dict, country_name):
    """
    Plot comparison of different regression models.
    
    Parameters
    ----------
    days : np.ndarray
        Days since first outbreak
    cumulative_cases : np.ndarray
        Cumulative cases (data points)
    models_dict : dict
        Dictionary of models from compare_models()
    country_name : str
        Name of the country
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot data points
    ax.scatter(days, cumulative_cases, color='black', marker='o', s=60, 
               label='Data points', zorder=4, alpha=0.7)
    
    # Generate smooth predictions
    days_pred = np.linspace(days.min(), days.max(), 200)
    
    # Plot each model
    colors = {'linear': 'red', 'polynomial_2': 'blue', 'polynomial_3': 'green'}
    linestyles = {'linear': '-', 'polynomial_2': '--', 'polynomial_3': '-.'}
    
    for key, value in models_dict.items():
        model = value['model']
        name = value['name']
        r2 = value['r2']
        
        if key == 'linear':
            pred = model.predict(days_pred.reshape(-1, 1))
        else:
            pred = model.predict(days_pred.reshape(-1, 1))
        
        ax.plot(days_pred, pred, 
               color=colors.get(key, 'gray'),
               linestyle=linestyles.get(key, '-'),
               linewidth=2.5,
               label=f"{name} (R² = {r2:.4f})",
               zorder=3)
    
    ax.set_xlabel('Days since first outbreak', fontsize=12)
    ax.set_ylabel('Cumulative Cases', fontsize=12)
    ax.set_title(f'Ebola Epidemic in {country_name}: Model Comparison', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()