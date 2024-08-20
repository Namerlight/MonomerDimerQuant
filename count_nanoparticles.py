import os
import pickle
import argparse
import cv2 as cv
import numpy as np
import pandas as pd
from flaml import AutoML
from time import time, sleep
from ultralytics import YOLO
from collections import Counter
from sahi.slicing import slice_image
from utils import compute_particle_data

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

model_path = os.path.join("models", "yolo_np_auns.pt")
model = YOLO(model_path)

def main(image_path, slice_size = None):

    total_time_start = time()

    image_file_name = image_path.split(os.path.sep)[-1].split(".")[0] + "_" + str(int(time()))
    bboxes_data_folder = os.path.join("output", image_file_name, "particle_data")

    if not os.path.exists("output"):
        os.makedirs(os.path.join("output"))

    im_folder = os.path.join("output", image_file_name, "processed")

    im = cv.imread(image_path)
    h, w = im.shape[:2]

    if not os.path.exists(im_folder):
        os.makedirs(im_folder)
        os.makedirs(bboxes_data_folder)

    if slice_size and h >= 2*slice_size and w >= 2*slice_size:
        slice_image_result = slice_image(
            image=image_path,
            output_file_name=image_path.split(os.path.sep)[-1].split(".")[0],
            output_dir=im_folder,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=0.0,
            overlap_width_ratio=0.0,
        )
    else:
        cv.imwrite(os.path.join(im_folder, image_path.split(os.path.sep)[-1]), im)

    sliced_images = [os.path.join(im_folder, filenames) for filenames in os.listdir(im_folder)]
    store_boxes = []

    sleep(1)

    tot_ct, tot_mono, tot_oli = 0, 0, 0

    for im_path in sliced_images:

        im_put = cv.imread(im_path)

        results = model.predict(source=im_put, conf=0.15, iou=0.2)
        result = results[0]

        print(f"Num of Boxes for {im_path}:, {len(result.boxes)}")

        store_boxes.append(len(result.boxes))

        # cv.imshow('Image to show', result.orig_img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        nums, x1s, x2s, y1s, y2s = [], [], [], [], []

        for n, box in enumerate(result.boxes):
            bxs = box.xyxy[0].cpu().numpy().tolist()

            x0, x1, y0, y1 = bxs[0], bxs[2], bxs[1], bxs[3]

            start_point, end_point = (int(x0), int(y0)), (int(x1), int(y1))
            cv.rectangle(result.orig_img, start_point, end_point, color=(255, 255, 255), thickness=1)

            cv.putText(img=result.orig_img, text=str(n), org=(int(x0), int(y0)),
                       fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 255, 255), thickness=1)

            nums.append(n), x1s.append(x0), y1s.append(y0), x2s.append(x1), y2s.append(y1)

        result.orig_img = cv.resize(result.orig_img, (512, 512), interpolation=cv.INTER_AREA)

        # cv.imshow('Image to show', result.orig_img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        final_sheet = {
            "image_path_from_utils": [im_path] * len(nums), "index": nums, "x1": x1s, "y1": y1s, "x2": x2s, "y2": y2s,
        }

        data_df = pd.DataFrame.from_dict(final_sheet)
        data_df.to_csv(os.path.join(bboxes_data_folder, im_path.split(os.path.sep)[-1].split(".")[0] + ".csv"), index=False)

    bboxes_data_files = [os.path.join(bboxes_data_folder, filenames) for filenames in os.listdir(bboxes_data_folder)]

    automl = AutoML()

    with open(os.path.join("models", "lgbm_np_auns.pkl"), "rb") as f:

        automl = pickle.load(f)

        for bbx_file in bboxes_data_files:
            boundingboxes_df = pd.read_csv(bbx_file)
            compute_feats = compute_particle_data.compute_features(bbx_df=boundingboxes_df)
            compute_feats.to_csv(bbx_file, index=False)

            compute_feats['width'] = compute_feats['x2'] - compute_feats['x1']
            compute_feats['height'] = compute_feats['y2'] - compute_feats['y1']
            compute_feats["radius"] = np.sqrt(((compute_feats["x2"] - compute_feats["x1"]) / 2) ** 2 + ((compute_feats["y2"] - compute_feats["y1"]) / 2) ** 2)

            test_x_test = compute_feats[
                ['radius', 'width', 'height', 'distance_closest_scalar', 'total_intensity', 'max_intensity',
                 'avg_intensity', 'sigma_x', 'sigma_y']]

            test_preds = automl.predict(test_x_test)
            test_preds = ["Oligomer" if res == 1 else "Monomer" for res in test_preds]

            compute_feats["predicted_state"] = test_preds

            compute_feats.to_csv(bbx_file, index=False)

            # print(f"Total Count for {bbx_file}: {len(test_preds)}, {list(Counter(test_preds).keys()), list(Counter(test_preds).values())}")

            tot_ct += len(test_preds)
            tot_mono += test_preds.count("Monomer")
            tot_oli += test_preds.count("Oligomer")

        final_result = (f'Total count of nanoparticles for {image_path}: {tot_ct}.\n'
                        f'Number of Monomers: {tot_mono}\n'
                        f'Number of Oligomers: {tot_oli}\n'
                        f'Total processing time: {time() - total_time_start} seconds.\n'
                        f'Results have been saved to {os.path.join("output", image_file_name, "")}. '
                        f'Please look at the \'particle_data\' folder for detailed particle data and results.')

        print(final_result)

        with open(os.path.join("output", image_file_name, "results.txt"), 'w') as rf:
            rf.write(final_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--image_path', type=str, required=True, help='Path to Image.')
    parser.add_argument('-s', '--slice_size', type=int, required=False, help='Size of slices')
    args = parser.parse_args()

    main(image_path=args.image_path, slice_size=args.slice_size)