import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from mw_plot import MWSkyMap
from sklearn.cluster import KMeans

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

