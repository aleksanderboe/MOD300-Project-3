import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from mw_plot import MWSkyMap

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
