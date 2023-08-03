import os
import cv2
import numpy as np
import pytesseract
from pytesseract import Output

import argparse
import json

# Feature Control Frame Symbol
symbol_map = {
    "⏤": 'Straightness',
    "⏥": 'Flatness', 
    "⌭": 'Cylindricity', 
    "○": 'Circularity', 
    "⌯": 'Symmetry', 
    "⌖": 'Position', 
    "◎": 'Concentricity', 
    "⟂": 'Perpendicularity', 
    "∥": 'Parallelism', 
    "∠": 'Angularity', 
    "⌓": 'Profile of a surface', 
    "⌒": 'Profile of a line',
    "⌰": 'Total run-out', 
    "↗": 'Circular run-out'
}
feature_symbol_map = {
    'Ⓕ': '(Free state)',
    'Ⓛ': '(LMC)',
    'Ⓜ': '(MMC)',
    'Ⓟ': '(Projected tolerance zone)',
    'Ⓢ': '(RFS)',
    'Ⓣ': '(Tangent plane)',
    'Ⓤ': '(Unequal bilateral)'
}


def image_preprocessing( image ):
    
    H, W, _ = image.shape
    
    gray_img = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )

    ret, threshold_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

    image = cv2.erode(threshold_img, kernel)     # 先侵蝕，將白色小圓點移除

    image = remove_the_blackborder(image)

    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print(len(contours))
    for obj in contours:
        perimeter = cv2.arcLength(obj, True)
        approx = cv2.approxPolyDP(obj, 0.02*perimeter, True)
        CornerNum = len(approx)
        x, y, w, h = cv2.boundingRect(approx)
        if CornerNum == 4:
            cut_image = image[ y: y+h, x: x+w ]
            # cv2.imshow("cut_image",cut_image)
            # cv2.waitKey(0)
            return cut_image

    return None

def Tess_OCR( image ):

    custom_config = r'-l eng --psm 6 --oem 3'
    data = pytesseract.image_to_data(image, output_type=Output.DICT, config=custom_config)

    for i in range(len(data["text"])):
        tmp_level = data["level"][i]
        text = data["text"][i]
        TEXT_str = ''
        if tmp_level == 5:
            for t in text:
                TEXT_str += t

    return TEXT_str

def remove_the_blackborder( image ):

    # img = cv2.medianBlur(image, 5) 
 
    edges_y, edges_x = np.where(image==0) ##h, w
    bottom = min(edges_y)             
    top = max(edges_y) 
    height = top - bottom            
                                   
    left = min(edges_x)           
    right = max(edges_x)             
    height = top - bottom 
    width = right - left

    res_image = image[bottom:bottom+height, left:left+width]

    return res_image


def main( dir_path ):

    details_json = {}
    files_list = next(os.walk(dir_path), (None, None, []))[2]
    img_files = [ img_file for img_file in files_list if not img_file.endswith('.json')] 
    for img_file in img_files:
        img_name = img_file.rsplit('.', 1)[0]
        datum_map = {}
        img = cv2.imread( os.path.join(dir_path, img_file) )
        image = image_preprocessing(img)

        datum = Tess_OCR( image )

        datum_map['datum'] = datum

        details_json[ img_name ] = datum_map


    json_path = os.path.join(dir_path, 'recognize.json')
    with open(json_path, 'w', encoding='utf-8') as outfile:
        json.dump(details_json, outfile)

    return


parser = argparse.ArgumentParser()

parser.add_argument('--path', help='Datum image files dir path')

opt = parser.parse_args()

if __name__ == '__main__':
    main( opt.path )