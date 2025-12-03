"""
Ebola data analysis functions.
Functions for loading, processing, and visualizing Ebola epidemic data.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from pathlib import Path


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


def load_all_countries_data(data_dir=None):
    """
    Load Ebola data for all three countries.
    
    Parameters
    ----------
    data_dir : str or Path, optional
        Directory containing the data files. If None, uses default location
        relative to this module.
        
    Returns
    -------
    countries_data : dict
        Dictionary with country names as keys and tuples (days, new_cases, cumulative_cases) as values
    """
    if data_dir is None:
        # Use data directory relative to this module
        module_path = Path(__file__).parent
        data_path = module_path / "data"
    else:
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

