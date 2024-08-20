import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import math

def preproc_image(img_path, threshold = 15, save_path = None) -> np.ndarray:

    img = cv2.imread(img_path)

    ret, img = cv2.threshold(img, threshold, 255, cv2.THRESH_TOZERO)
    img[img > 10] *= 3
    # img = cv2.fastNlMeansDenoising(src=img, dst=None, h=15, templateWindowSize=5, searchWindowSize=15)
    # img = cv2.blur(img, (5, 5))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if save_path:
        cv2.imwrite(save_path, img)

    return img

def plot_heightmap(img, angle = None, save_path = None) -> None:

    # histg = cv2.calcHist([img], [0], None, [256], [1, 256])

    fig = plt.figure(figsize=(12, 15))
    ax0 = fig.add_subplot(211)
    ax1 = fig.add_subplot(212, projection='3d')

    X = np.arange(0, img.shape[1], 1)
    Y = np.arange(0, img.shape[0], 1)
    X, Y = np.meshgrid(Y, X)
    Z = np.transpose(img)

    cmap = matplotlib.colormaps["plasma"].copy()
    cmap.set_under(color='black')
    ax1.set_xlabel('Pixel Coordinate')
    ax1.set_ylabel('Pixel Coordinate', labelpad=10)
    ax1.set_zlabel('Intensity')

    aspects = math.gcd(img.shape[0], img.shape[1])
    x_scale, y_scale = img.shape[1]/aspects, img.shape[0]/aspects

    surf = ax1.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=False, vmin=20)
    ax1.set_box_aspect(aspect=(y_scale*3, x_scale*3, x_scale/1.5))

    if angle:
        ax1.view_init(angle[0], angle[1])

    fig.tight_layout()

    im = ax0.imshow(img)

    plt.show()


def main(image_path=None):

    if not image_path:
        image_path = "../data/uncorrelated/optical_uncorrelated/001-1-2s.jpeg"

    simg = cv2.imread(image_path)

    cv2.imshow('Image to show', simg)
    cv2.waitKey(0)

    processed_img = preproc_image(
        img_path=image_path,
        threshold=0,
    )

    cv2.imshow('Image to show', processed_img)
    cv2.waitKey(0)

    plot_heightmap(
        img=processed_img,
        angle=(30, 0)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--image_path', type=str, required=True, help='Path to Image.')
    args = parser.parse_args()

    main(image_path=args.image_path)

