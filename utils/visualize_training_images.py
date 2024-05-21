import os
import json
import cv2 as cv
import numpy as np
import pandas as pd

np.set_printoptions(threshold=np.inf)
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.width = None
pd.options.display.max_colwidth = None

def pandafy_bbboxes(path_to_bbox_file: str) -> pd.DataFrame:
    """
    Gets bounding boxes from a json file. The json file is in YOLO format. This is for a single image.

    Args:
        path_to_bbox_file: path to json file

    Returns: A Dataframe with rows for each box's coordinates and label.
    """
    with open(path_to_bbox_file) as f:
        json_data = json.load(f)
        bbx_df = pd.json_normalize(json_data.get('boxes'))
        bbx_df["image_path_from_utils"] = os.path.join(os.sep.join(path_to_bbox_file.split(os.sep)[:-2]), "optical_images", json_data.get("key"))
        bbx_df["x1"] = (bbx_df["x"].astype(int) - (bbx_df["width"].astype(int))/2)
        bbx_df["x2"] = (bbx_df["x"].astype(int) + (bbx_df["width"]).astype(int)/2)
        bbx_df["y1"] = (bbx_df["y"].astype(int) - (bbx_df["height"]).astype(int)/2)
        bbx_df["y2"] = (bbx_df["y"].astype(int) + (bbx_df["height"]).astype(int)/2)
        bbx_df = bbx_df.reindex(columns=['image_path_from_utils', "x1", "y1", "x2", "y2", "label"])
        return bbx_df

def show_image_with_bboxes(bbx_df: pd.DataFrame):
    """
    Draws bboxes in image from bounding boxes from a pandas dataframe (call pandafy and pass the output dataframe here).

    Args:
        bbx_df: pd dataframe with image name, x and y, width and height, and label per box
    """
    if len(list(bbx_df["image_path_from_utils"].unique())) > 1:
        raise RuntimeError("This function only works for a single image per dataframe input.")

    image_path = bbx_df["image_path_from_utils"].unique()[0]

    img = cv.imread(image_path)
    cv.imshow('Image without BBoxes', img)
    cv.waitKey(0)

    for row in bbx_df.itertuples():
        color = (0, 75, 255) if row.label == 'monomer' else (255, 150, 0)
        cv.rectangle(img=img, pt1=(int(row.x1), int(row.y1)), pt2=(int(row.x2), int(row.y2)), color=color, thickness=1)
        cv.putText(img=img, text=str(row.Index), org=(int(row.x1), int(row.y1)), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=color, thickness=1)

    cv.imshow('Image with BBoxes', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    bbox_data_dir = os.path.join("..", "data", "correlated", "bboxes_two_classes")
    bbox_data_files = [os.path.join(bbox_data_dir, filenames) for filenames in os.listdir(bbox_data_dir)]

    for file in bbox_data_files:
        boundingboxes_df = pandafy_bbboxes(path_to_bbox_file=file)
        show_image_with_bboxes(bbx_df=boundingboxes_df)
