import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from mw_plot import MWSkyMap
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


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
    plt.title(f"{center} - Radius: {round(radius[0], 2)} arcsec")
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

    Parameters
    ----------
    kmeans: sklearn.cluster.KMeans
        Trained KMeans model

    Returns
    -------
    descriptions: list
        List of strings describing each cluster
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

# Topic 2, Task 2: Better fitting functions for Ebola data
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

# Topic 2: Task 3
def train_neural_network(days, cumulative_cases, hidden_layer_sizes=(32,32), test_size=0.2, random_state=0): 
    """
    Train neural network 

    Parameters
    ----------
    days: np.ndarray
        Days since first outbreak 
    cumulative_cases: np.adarray
        Cumulative cases
    hidden_layer_size: tuple
        Sizes of hidden layers in the MLP
    test_size: float
        Fraction of data to keep for testing 
    random_state: int
        For reproducibility

    Returns
    -------

    model: sklearn.neural_network.MLPRegressor
    x_scaler: 
    y_scaler: 
    X_train, X_test, y_train, y_test: np.adarray
        Train/test splits in original units 
    metrics: dict 
        R² and RMSE for train and test sets 
    """
    X = days.reshape(-1,1).astype(float)
    y = cumulative_cases.astype(float)

    n = len(X)
    n_test = int(np.ceil(test_size * n))
    n_train = n - n_test

    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation="relu", solver="adam", max_iter=5000, random_state=random_state)
    model.fit(X_train_scaled, y_train_scaled)

    def predict_inverse(X_scaled): 
        y_scaled_pred = model.predict(X_scaled)
        return y_scaler.inverse_transform(y_scaled_pred.reshape(-1, 1)).ravel()
    
    y_train_pred = predict_inverse(X_train_scaled)
    y_test_pred = predict_inverse(X_test_scaled)

    metrics = {
        "train_r2": r2_score(y_train, y_train_pred), 
        "test_r2": r2_score(y_test, y_test_pred),
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)), 
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)), 
    }

    return model, x_scaler, y_scaler, X_train, X_test, y_train, y_test, metrics

def plot_neural_network_prediction(days, cumulative_cases, model, x_scaler, y_scaler, country_name, ax=None): 
    """
    Plot data points and NN prediction curve over the whole time range. 

    Parameters
    ----------
    days: np.ndarray
        Days since first outbreak
    cumulative_cases: np.adarray
        Cumulative cases
    model: sklearn.neural_network.MLPRegressor
        Trained neural network model
    x_scaler: sklearn.preprocessing.StandardScaler
        Scaler used for normalizing the input data
    y_scaler: sklearn.preprocessing.StandardScaler
        Scaler used for normalizing the output data
    country_name: str
        Name of the country
    ax: matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    """
    if ax is None: 
        fig, ax = plt.subplots(figsize=(10, 6)) 
    
    ax.scatter(days, cumulative_cases, color="black", s=50, label="Data points", zorder=3)

    days_grid = np.linspace(days.min(), days.max(), 200).reshape(-1, 1)
    days_grid_scaled = x_scaler.transform(days_grid)

    y_grid_scaled = model.predict(days_grid_scaled)
    y_grid = y_scaler.inverse_transform(y_grid_scaled.reshape(-1, 1)).ravel()

    ax.plot(days_grid.ravel(), y_grid, linewidth=2.5, label="Neural network prediction", color="purple", zorder=2)

    ax.set_xlabel("Days since first outbreak", fontsize=12)
    ax.set_ylabel("Cumulative cases", fontsize=12)
    ax.set_title(f"Ebola epidemic in {country_name}: Neural Network Model", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)

    return ax

def run_task3_neural_networks(test_size=0.2, hidden_layer_sizes=(32, 32)): 
    """
    Train a neural network for each country and plot predictions. 

    Parameters
    ----------
    test_size: float
        Fraction of data to keep for testing 
    hidden_layer_sizes: tuple
        Sizes of hidden layers in the MLP
    """

    countries_data = load_all_countries_data()

    for country, (days, new_cases, cumulative_cases) in countries_data.items(): 
        print(f"\n{'='*60}")
        print(f"Task 3 - Neural network for {country}")
        print(f"{'='*60}")

        (model, x_scaler, y_scaler, 
         X_train, X_test, 
         y_train, y_test, metrics) = train_neural_network(days, cumulative_cases, hidden_layer_sizes=hidden_layer_sizes, test_size=test_size, random_state=0)
        
        print("Metrics: ")
        for k, v in metrics.items(): 
            print(f"{k:12s}: {v:.4f}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_neural_network_prediction(days, cumulative_cases, model, x_scaler, y_scaler, country_name=country, ax=ax)

        plt.show()

def create_sequences(data, n_steps=3):
    """
    Create sequences for LSTM training.
    
    Parameters
    ----------
    data : np.ndarray
        1D array of time series values
    n_steps : int
        Number of time steps to use as input (lookback window)
        
    Returns
    -------
    X : np.ndarray
        Input sequences of shape (n_samples, n_steps, 1)
    y : np.ndarray
        Target values of shape (n_samples,)
    """
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data) - 1:
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def train_lstm_model(days, cumulative_cases, n_steps=3, test_size=0.2, epochs=100, batch_size=1, verbose=0):
    """
    Train an LSTM model for time series prediction.
    
    Parameters
    ----------
    days : np.ndarray
        Days since first outbreak
    cumulative_cases : np.ndarray
        Cumulative cases over time
    n_steps : int
        Number of time steps to look back (sequence length)
    test_size : float
        Fraction of data to use for testing (taken from the end, as is proper for time series)
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    verbose : int
        Verbosity level (0=silent, 1=progress bar)
        
    Returns
    -------
    model : keras.Model
        Trained LSTM model
    scaler : sklearn.preprocessing.StandardScaler
        Scaler fitted on training data
    X_train, X_test, y_train, y_test : np.ndarray
        Train/test sequences
    metrics : dict
        R² and RMSE for train and test sets
    """

    
    data = cumulative_cases.astype(np.float32)
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).ravel()
    
    X, y = create_sequences(data_scaled, n_steps=n_steps)
    
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    # Train model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=False)
    
    # Make predictions
    y_train_pred_scaled = model.predict(X_train, verbose=0)
    y_test_pred_scaled = model.predict(X_test, verbose=0)
    
    y_train_pred = scaler.inverse_transform(y_train_pred_scaled).ravel()
    y_test_pred = scaler.inverse_transform(y_test_pred_scaled).ravel()
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1)).ravel()
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    
    metrics = {
        "train_r2": r2_score(y_train_actual, y_train_pred),
        "test_r2": r2_score(y_test_actual, y_test_pred),
        "train_rmse": np.sqrt(mean_squared_error(y_train_actual, y_train_pred)),
        "test_rmse": np.sqrt(mean_squared_error(y_test_actual, y_test_pred)),
    }
    
    return model, scaler, X_train, X_test, y_train, y_test, metrics

def predict_future_lstm(model, scaler, last_sequence, n_future):
    """
    Predict future values using LSTM model.
    
    Parameters
    ----------
    model : keras.Model
        Trained LSTM model
    scaler : sklearn.preprocessing.StandardScaler
        Scaler used for normalization
    last_sequence : np.ndarray
        Last n_steps values (scaled) to use as starting point
    n_future : int
        Number of future time steps to predict
        
    Returns
    -------
    predictions : np.ndarray
        Predicted future values (in original scale)
    """
    predictions = []
    current_seq = last_sequence.copy()
    
    for _ in range(n_future):
        X_input = current_seq.reshape((1, len(current_seq), 1))
        pred_scaled = model.predict(X_input, verbose=0)
        predictions.append(pred_scaled[0, 0])
        current_seq = np.append(current_seq[1:], pred_scaled[0, 0])
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).ravel()
    return predictions

def plot_lstm_prediction(days, cumulative_cases, model, scaler, n_steps=3, country_name="", ax=None):
    """
    Plot LSTM predictions over the data range and into the future.
    
    Parameters
    ----------
    days : np.ndarray
        Days since first outbreak
    cumulative_cases : np.ndarray
        Cumulative cases (actual data)
    model : keras.Model
        Trained LSTM model
    scaler : sklearn.preprocessing.StandardScaler
        Scaler used for normalization
    n_steps : int
        Number of time steps used in sequences
    country_name : str
        Name of the country
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.scatter(days, cumulative_cases, color='black', marker='o', s=50, 
               label='Actual data', zorder=3, alpha=0.7)
    
    data_scaled = scaler.transform(cumulative_cases.reshape(-1, 1)).ravel()
    X, y = create_sequences(data_scaled, n_steps=n_steps)
    
    X_all = X.reshape((X.shape[0], X.shape[1], 1))
    y_pred_scaled = model.predict(X_all, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled).ravel()
    
    days_pred = days[n_steps:n_steps+len(y_pred)]
    
    ax.plot(days_pred, y_pred, 'b-', linewidth=2, 
            label='LSTM prediction (fitted)', zorder=2, alpha=0.8)
    
    n_future = 30  # Predict 30 days into the future
    last_seq = data_scaled[-n_steps:]
    future_pred = predict_future_lstm(model, scaler, last_seq, n_future)
    future_days = np.arange(days[-1] + 1, days[-1] + 1 + n_future)
    
    ax.plot(future_days, future_pred, 'r--', linewidth=2, 
            label=f'LSTM prediction (future, {n_future} days)', zorder=2, alpha=0.8)
    
    ax.set_xlabel('Days since first outbreak', fontsize=12)
    ax.set_ylabel('Cumulative cases', fontsize=12)
    ax.set_title(f'Ebola epidemic in {country_name}: LSTM Model Prediction', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    return ax

def run_task4_lstm(n_steps=3, test_size=0.2, epochs=100):
    """
    Train LSTM models for all countries and visualize predictions.
    
    Parameters
    ----------
    n_steps : int
        Number of time steps for LSTM lookback window
    test_size : float
        Fraction of data for testing
    epochs : int
        Number of training epochs
    """
    countries_data = load_all_countries_data()
    
    for country, (days, new_cases, cumulative_cases) in countries_data.items():
        print(f"\n{'='*60}")
        print(f"LSTM model for {country}")
        print(f"{'='*60}")
        
        try:
            (model, scaler, X_train, X_test, 
             y_train, y_test, metrics) = train_lstm_model(
                days, cumulative_cases, 
                n_steps=n_steps, 
                test_size=test_size, 
                epochs=epochs,
                verbose=0
            )
            
            print("Metrics:")
            for k, v in metrics.items():
                print(f"  {k:12s}: {v:.4f}")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_lstm_prediction(days, cumulative_cases, model, scaler, 
                                n_steps=n_steps, country_name=country, ax=ax)
            plt.show()
            
        except Exception as e:
            print(f"Error training LSTM for {country}: {e}")
            import traceback
            traceback.print_exc()