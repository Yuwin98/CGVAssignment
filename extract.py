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


row = []
column = []

# Sorting boxes to their respective row and columns
for i in range(len(box)):
    if i == 0:
        column.append(box[i])
        previous = box[i]
    else:
        if box[i][1] <= previous[1] + meanHeight / 2:
            column.append(box[i])
            previous = box[i]
            if i == len(box) - 1:
                row.append(column)
        else:
            row.append(column)
            column = []
            previous = box[i]
            column.append(box[i])

# No of Cells
countCol = 0
row = row[3:]

for i in range(len(row)):
    countCol = len(row[i])
    if countCol > countCol:
        countCol = countCol
    center = [int(row[i][j][0] + row[i][j][2] / 2) for j in range(len(row[i])) if row[0]]
    center = np.array(center)
    center.sort()

finalboxes = []
for i in range(len(row)):
    lis = []
    for k in range(countCol):
        lis.append([])
    for j in range(len(row[i])):
        diff = abs(center - (row[i][j][0] + row[i][j][2] / 4))
        minimum = min(diff)
        indexing = list(diff).index(minimum)
        lis[indexing].append(row[i][j])
    finalboxes.append(lis)

signatures = []
padded_signatures = []
idx = 0
for i in range(len(row)):
    sigImg = finalboxes[i][4].pop()
    finalboxes[i] = finalboxes[i][1:-1]
    y, x, w, h = sigImg[0], sigImg[1], sigImg[2], sigImg[3]
    signature = img_cpy[x:x + h, y:y + w]
    padded_signature = resize_with_padding(signature)
    cv2.imwrite(path + f'sig_{idx}.jpeg', signature)
    cv2.imwrite(path + f'sig_pad_{idx}.jpeg', padded_signature)
    signatures.append(signature)
    padded_signatures.append(padded_signature)
    idx += 1

outer = []
idx = 0
print(finalboxes[0])
final_box_with_text = []
for i in range(len(finalboxes)):
    for j in range(len(finalboxes[i])):
        for k in range(len(finalboxes[i][j])):
            y, x, w, h = finalboxes[i][j][k][0], finalboxes[i][j][k][1], finalboxes[i][j][k][2], \
                         finalboxes[i][j][k][3]
            finalimg = img_cpy[x:x + h, y:y + w - 5]
            out = pytesseract.image_to_string(finalimg)
            if len(out) > 5:
                final_str = out.strip()
                outer.append(final_str)
            else:
                out = pytesseract.image_to_string(finalimg, config='--psm 12')
                pat = re.compile('(Mr|Ms)')
                final_str = pat.findall(out)[0]
                outer.append(final_str)
            final_box_with_text.append(str(final_str))
            cv2.imwrite(path + f'table_cell_{idx}.jpeg', finalimg)
            idx += 1

data = []
size = 3
for i in range(len(row)):
    studentId = outer[i * size]
    title = outer[i * size + 1][:] if len(outer[i * size + 1]) > 0 else ''
    name = outer[i * size + 2]
    student = (i + 1, studentId, title, name)
    data.append(student)

return data, signatures, final_box_with_text


def resize_with_padding(img):
    img = image_resize(img, width=416)
    new_image_width = 416
    new_image_height = 416
    color = (0, 0, 0)
    result = np.full((new_image_height, new_image_width, 3), color, dtype=np.uint8)
    old_image_height, old_image_width, channels = img.shape
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2
    result[y_center:y_center + old_image_height, x_center:x_center + old_image_width] = img
    return result


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

