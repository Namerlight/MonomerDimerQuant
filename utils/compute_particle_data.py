import os
import math
import torch

import cv2 as cv
import numpy as np
import pandas as pd
from pprint import pprint
from scipy.spatial.distance import cdist, euclidean
from .visualize_training_images import pandafy_bbboxes_from_json


def compute_center_points(bbx_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the center points of each bounding box in the dataframe.
    Args:
        bbx_df: dataframe with bounding boxes

    Returns: dataframe with new columns for x and y centerpoints
    """

    bbx_df["x_center"] = (bbx_df["x1"] + bbx_df["x2"]) / 2
    bbx_df["y_center"] = (bbx_df["y1"] + bbx_df["y2"]) / 2

    return bbx_df


def compute_closest_particle_info(bbx_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the nearest particle for each particle in the dataframe.
    Args:
        bbx_df: dataframe with bounding boxes

    Returns: dataframe with new columns for the x, y and Euclidean distances to nearest as well as index of nearest
    particle
    """

    dist_matrix = cdist(bbx_df[['x_center', 'y_center']], bbx_df[['x_center', 'y_center']])
    closest_indices = dist_matrix.argsort(axis=1)[:, 1]
    closest_coordinates = bbx_df.iloc[closest_indices]

    bbx_df['closest_x'] = closest_coordinates['x_center'].values
    bbx_df['closest_y'] = closest_coordinates['y_center'].values

    # Iterrows is nominally bad pandas, but when I'm extracting and placing individual row indexes
    # it seems prudent to go for more straightforward, less-likely-to-bug-out code.
    for idx, row in bbx_df.iterrows():
        bbx_df.at[idx, "closest_idx"] = closest_indices[idx]
        bbx_df.at[idx, "distance_closest_scalar"] = euclidean((row['x_center'], row['y_center']),
                                                              (row['closest_x'], row['closest_y']))
    return bbx_df


def intensity_op(row, full_img: np.ndarray, to_calc: str):
    cropped_image = full_img[int(row["y1"]):int(row["y2"]), int(row["x1"]):int(row["x2"])]
    gray_image = cv.cvtColor(cropped_image, cv.COLOR_BGR2GRAY)
    calcs_dicts = {
        "total": np.sum(gray_image),
        "max": np.max(gray_image),
        "avg": np.sum(gray_image) / (int(row["x2"] - row["x1"]) * int(row["y2"] - row["y1"]))
    }
    return calcs_dicts.get(to_calc)


def compute_intensity_info(bbx_df: pd.DataFrame, full_img: np.ndarray) -> pd.DataFrame:
    """
    Computes the intensity information per particle by loading the image and cropping it to each bounding box in turn.

    Args:
        bbx_df: dataframe with bounding boxes
        full_img: the original full image from which we'll crop individual boxes

    Returns: dataframe with new columns for the total, average and max intensity per particle.
    """

    bbx_df["total_intensity"] = bbx_df.apply(intensity_op, full_img=full_img, to_calc="total", axis=1)
    bbx_df["max_intensity"] = bbx_df.apply(intensity_op, full_img=full_img, to_calc="max", axis=1)
    bbx_df["avg_intensity"] = bbx_df.apply(intensity_op, full_img=full_img, to_calc="avg", axis=1)

    return bbx_df


def gen_gaussian(x_l: int, y_l: int, sigma_x: torch.tensor, sigma_y: torch.tensor, max_val: None) -> torch.tensor:
    """
    Generates a 2D gaussian based on the specified values

    Args:
        x_l: bbox width of the particle we're generating the PDF for
        y_l: bbox height of the particle image we're generating the PDF for
        sigma_x: standard dev in the x-axis. Parameterized by grad desc
        sigma_y: standard dev in the y-axis. Parameterized by grad desc
        max_val: maximum pixel intensity of the original particle, used to scale the output

    Returns: torch tensor array containing a 2D gaussian meant to emulate a particle

    """

    xc, yc = x_l // 2, y_l // 2
    x = torch.arange(start=0, end=x_l, dtype=float)
    y = torch.arange(start=0, end=y_l, dtype=float)[:, None]
    x, y = x-xc, y-yc

    exp_part = x ** 2 / (2 * sigma_x ** 2) + y ** 2 / (2 * sigma_y ** 2)
    gss = 1 / (2 * torch.Tensor([math.pi]) * sigma_x * sigma_y) * torch.exp(-exp_part)
    normalized = (gss - torch.min(gss))/(torch.max(gss) - torch.min(gss))

    if max_val:
        normalized = torch.clamp(normalized, max=max_val)

    return normalized


def compute_point_density_function(bbx_df: pd.DataFrame, full_img: np.ndarray) -> pd.DataFrame:
    """
    Computes an approximated point density function for each row by carrying out gradient descent and backprop on
    a 2D gaussian function using a pixelwise MSE loss.

    Args:
        bbx_df: dataframe with bounding boxes
        full_img: the original full image from which we'll crop individual boxes

    Returns: dataframe with new columns for the mean, standard deviation and variance
    """

    for idx, row in bbx_df.iterrows():
        cropped_image = full_img[int(row["y1"]):int(row["y2"]), int(row["x1"]):int(row["x2"])]
        gray_image = cv.cvtColor(cropped_image, cv.COLOR_BGR2GRAY)

        sgx, sgy = torch.tensor(1.0, requires_grad=True), torch.tensor(1.0, requires_grad=True)
        learning_rate, n_iter = 0.01, 100

        for i in range(n_iter):
            # making predictions with forward pass
            y_pred = gen_gaussian(
                x_l=(int(row["x2"])-int(row["x1"])),
                y_l=(int(row["y2"])-int(row["y1"])),
                sigma_x=sgx, sigma_y=sgy, max_val=row["max_intensity"]/255
            )
            loss = torch.mean((y_pred - torch.from_numpy(gray_image)) ** 2)
            loss.backward()
            sgx.data -= learning_rate * sgx.grad.data
            sgy.data -= learning_rate * sgy.grad.data
            sgx.grad.data.zero_(), sgy.grad.data.zero_()

        bbx_df.at[idx, "sigma_x"] = sgx.detach().item()
        bbx_df.at[idx, "sigma_y"] = sgy.detach().item()

    return bbx_df


def compute_features(bbx_df: pd.DataFrame):
    """
    Computes particle intensity features from a pandas dataframe that inputs the x and y coordinates of the bboxes.

    Args:
        bbx_df: a DataFrame containing bboxes for one image
    """

    if len(list(bbx_df["image_path_from_utils"].unique())) > 1:
        raise RuntimeError("This function only works for a single image per dataframe input.")

    image_path = bbx_df["image_path_from_utils"].unique()[0]
    full_img = cv.imread(image_path)

    bbx_df = compute_center_points(bbx_df=bbx_df)
    bbx_df = compute_closest_particle_info(bbx_df=bbx_df)
    bbx_df = compute_intensity_info(bbx_df=bbx_df, full_img=full_img)
    bbx_df = compute_point_density_function(bbx_df=bbx_df, full_img=full_img)

    # Uncomment to print features before returning
    # pprint(bbx_df)

    return bbx_df


# if __name__ == '__main__':
#
#     save = True
#
#     bbox_data_dir = os.path.join("..", "data", "correlated", "bboxes_two_classes")
#     bbox_data_files = [os.path.join(bbox_data_dir, filenames) for filenames in os.listdir(bbox_data_dir)]
#
#     for file in bbox_data_files:
#
#         boundingboxes_df = pandafy_bbboxes_from_json(path_to_bbox_file=file)
#         compute_feats = compute_features(bbx_df=boundingboxes_df)
#
#         if save:
#             pth_cmps = file.split(os.path.sep)
#
#             try: os.mkdir(os.path.join(os.path.sep.join(pth_cmps[:3]), "particle_data"))
#             except FileExistsError: pass
#
#             save_path = os.path.join(os.path.sep.join(pth_cmps[:3]), "particle_data", pth_cmps[-1].split(".")[0] + ".csv")
#             compute_feats.to_csv(save_path, index=True)