import numpy as np
import cv2 as cv
import scipy.stats as st
from pprint import pprint
import matplotlib.pyplot as plt

def gen_gaussian(x_l: int, y_l: int, sigma_x: float, sigma_y: float) -> np.ndarray:
    """
    Generates a 2D gaussian based on the specified values

    Args:
        x_l: bbox width of the particle we're generating the PDF for
        y_l: bbox height of the particle image we're generating the PDF for
        sigma_x: standard dev in the x-axis. Parameterized by grad desc
        sigma_y: standard dev in the y-axis. Parameterized by grad desc

    Returns: numpy array containing a

    """

    xc, yc = x_l // 2, y_l // 2
    x = np.arange(0, x_l, dtype=float)
    y = np.arange(0, y_l, dtype=float)[:, np.newaxis]
    x, y = x-xc, y-yc

    exp_part = x ** 2 / (2 * sigma_x ** 2) + y ** 2 / (2 * sigma_y ** 2)
    gss = 1 / (2 * np.pi * sigma_x * sigma_y) * np.exp(-exp_part)
    normalized = (gss - np.min(gss))/(np.max(gss) - np.min(gss))

    return normalized




# Press the green button in the gutter to run the script.
if __name__ == '__main__':



