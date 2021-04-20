import cv2
import numpy as np


def _remove_noise_and_smooth(image):
    # closing and opening
    kernel1 = np.zeros((5, 5), np.uint8)
    kernel2 = np.zeros((1, 1), np.uint8)
    dn_img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel1)
    dn_img = cv2.morphologyEx(dn_img, cv2.MORPH_OPEN, kernel1)

    # smoothing : GaussianBlur, bilateralFilter
    smooth_img = cv2.bilateralFilter(dn_img,9,75,75)
    return smooth_img


def _threshold(image):
    # threshold: adaptiveThreshold
    ret, th_img = cv2.threshold(image, 0, 255, cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
    return th_img


def _crop_ROI(image, x, y, w, h):
    cropped_image = image[ y:y+h , x:x+w ]
    return cropped_image


def _crop_border(image, border_offset=10, save=False):
    height = image.shape[0] - border_offset*2
    width = image.shape[1] - border_offset*2
    cropped_image = image[border_offset:border_offset+height, border_offset:border_offset+width]
    if save:
        cv2.imwrite(args.pre_save_path+'/border_crop_img.jpg', image)
    return cropped_image


def _rescale(image, scale_percent=110, save=False):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    rescaled_image = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
    if save:
        cv2.imwrite(args.pre_save_path+'/2_resized_img.jpg', rescaled_image)
    return rescaled_image
