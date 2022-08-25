import cv2
import numpy as np
import tensorflow as tf
from pytesseract import pytesseract
import re

from sortcontours import sort_contours
from deskewimg import deskewImg


def extractImg(image):
    path = './static/'

    # Save Copy of Original to Static Folder
    cv2.imwrite(path + 'original.jpeg', image)

    # Deskew Image and Save to Static Folder
    image = deskewImg(image)
    cv2.imwrite(path + 'deskew.jpeg', image)

    # Make a copy of original image
    img_cpy = image.copy()
    # Make a copy of image to draw contours
    img_cnt = image.copy()

    # Grayscale Image and Save to Static Folder
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(path + 'gray.jpeg', image)

    # Get the Binary Image and Save to Static Folder
    thresh, img_bin = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite(path + 'binary.jpeg', img_bin)

    # Invert the Binary Image and Save to SF
    img_bin = 255 - img_bin
    cv2.imwrite(path + 'binary_inv.jpeg', img_bin)

    # Create Horizontal and Vertical Kernels to detect Horizontal and vertical lines of the table
    kernel_len = np.array(image).shape[1] // 100
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # Detect vertical lines and Save to SF
    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
    cv2.imwrite(path + 'vl.jpeg', vertical_lines)

    # Detect Horizontal lines and Save to SF
    image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
    cv2.imwrite(path + 'hl.jpeg', horizontal_lines)

    # Combine both horizontal and vertical lines extracted from the image and save to SF
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    cv2.imwrite(path + 'vh.jpeg', img_vh)

    # Threshold the image and save to SF
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    thresh, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite(path + 'vh_thresh.jpeg', img_vh)

    # Find all contours in the image
    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, boundingBoxes = sort_contours(contours, method='top-to-bottom')

    # Set mean height (Approximated manually) - Approximated Number
    meanHeight = 60

    # Array of boxes to store extracted rectangles
    box = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 1500 > w > 100 and h > 50:
            box.append([x, y, w, h])
            cv2.rectangle(img_cnt, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Save image with drawn contours
    cv2.imwrite(path + 'image_contours.jpeg', img_cnt)
