import os

import cv2
import pytesseract
from flask import Flask, request, url_for, jsonify

from extract import extractImg

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

UPLOAD_FOLDER = './upload'
STATIC_FOLDER = './static'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER


@app.route('/checkattendance', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({
                'success': False
            })
        else:
            file = request.files['file']
            path = os.path.join(app.config['UPLOAD_FOLDER'], 'img.jpeg')
            file.save(path)
            image = cv2.imread(path)
            data, signatures, tableCellLabels = extractImg(image)

            original_image = getStaticUrl('original')
            deskew_image = getStaticUrl('deskew')
            grayscale_image = getStaticUrl('gray')
            binary_image = getStaticUrl('binary')
            binary_inv_image = getStaticUrl('binary_inv')
            vl_img = getStaticUrl('vl')
            hl_img = getStaticUrl('hl')
            vh_img = getStaticUrl('vh')
            vh_binary = getStaticUrl('vh_thresh')
            img_cnt = getStaticUrl('image_contours')

            tableCellImageList = createTableCellImageList(tableCellLabels)
            sigUrlList = createSignatureImgURLs(signatures)

            return dataToJson(data,
                              sigUrlList,
                              original_image,
                              deskew_image,
                              grayscale_image,
                              binary_image,
                              binary_inv_image,
                              vl_img,
                              hl_img,
                              vh_img,
                              vh_binary,
                              img_cnt,
                              tableCellImageList)


def createTableCellImageList(tableCellLabels):
    cellImagesWithLabels = []
    idx = 0
    for label in tableCellLabels:
        imgUrl = getStaticUrl(f'table_cell_{idx}')
        cellImagesWithLabels.append([label, imgUrl])
        idx += 1
    return cellImagesWithLabels

def createSignatureImgURLs(signatures):
    sigUrlList = []
    for i in range(len(signatures)):
        fullUrl = getStaticUrl(f'sig_{i}')
        sigUrlList.append(fullUrl)
    return sigUrlList


def getStaticUrl(filename, ext='jpeg'):
    host = 'http://127.0.0.1:5000'
    sigUrl = url_for('static', filename=f'{filename}.{ext}')
    return host + sigUrl


def dataToJson(data,
               signature_url_list,
               original_image,
               deskew_image,
               grayscale_image,
               binary_image,
               binary_inv_image,
               vl_img,
               hl_img,
               vh_img,
               vh_binary,
               img_cnt,
               table_cell_image_list
               ):
    return jsonify({
        'students': data,
        'signatures': signature_url_list,
        'original_img': original_image,
        'deskew_img': deskew_image,
        'grayscale_img': grayscale_image,
        'binary_img': binary_image,
        'binary_inv_img': binary_inv_image,
        'vl_img': vl_img,
        'hl_img': hl_img,
        'vh_img': vh_img,
        'vh_binary': vh_binary,
        'img_cnt': img_cnt,
        'table_cells': table_cell_image_list
    })


if __name__ == '__main__':
    app.run()