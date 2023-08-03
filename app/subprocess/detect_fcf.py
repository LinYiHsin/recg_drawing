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
    
    H, W = image.shape
    # dim = (W, H)
    # img_resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    side_length = int(max(W, H) * 1.5 )
    y_offset, x_offset = int((side_length-H)/2), int((side_length-W)/2)
    expand = np.zeros( (side_length, side_length), np.uint8 )
    expand[ y_offset : y_offset+H, x_offset : x_offset+W ] = image.copy()
    return_image = remove_the_blackborder(expand)

    return return_image

def Tess_OCR( part , image ):
    if part == 'symbol':
        custom_config = r'-l eng_gdt --psm 6 --oem 3'
    else:
        custom_config = r'-l eng_math --psm 8 --oem 3'
    data = pytesseract.image_to_data(image, output_type=Output.DICT, config=custom_config)

    TEXT_str = ''
    for i in range(len(data["text"])):
        tmp_level = data["level"][i]
        text = data["text"][i]
        TEXT = []
        if(tmp_level == 5):
            for t in text:
                if part == 'symbol':
                    if t in symbol_map:
                        TEXT.append(symbol_map[t])
                    else:
                        TEXT.append(t)
                else:
                    if t in feature_symbol_map:
                        TEXT.append(feature_symbol_map[t])
                    else:
                        TEXT.append(t)
            TEXT_str += ''.join(TEXT)

    return TEXT_str

def remove_the_blackborder( image ):

    img = cv2.medianBlur(image, 5) 
 
    edges_y, edges_x = np.where(img==255) ##h, w
    bottom = min(edges_y)             
    top = max(edges_y) 
    height = top - bottom            
                                   
    left = min(edges_x)           
    right = max(edges_x)             
    height = top - bottom 
    width = right - left

    res_image = image[bottom:bottom+height, left:left+width]

    return res_image

def get_coordinate( image ):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #二值化
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -5)
    # cv2.imshow("二值化圖片：", binary) #展示圖片
    # cv2.waitKey(0)
    # cv2.imwrite( os.path.join( output_folder, 'binary.png' ), binary )

    binary_inv = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, -5)
    # cv2.imwrite( os.path.join( output_folder, 'binary_inv.png' ), binary_inv )

    rows,cols=binary.shape
    scale = 4
    #識別橫線
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT,(cols//scale,1))
    eroded = cv2.erode(binary,kernel,iterations = 1)
    dilatedcol = cv2.dilate(eroded,kernel,iterations = 1)
    # cv2.imshow("表格橫線展示：",dilatedcol)
    # cv2.waitKey(0)
    # cv2.imwrite( os.path.join( output_folder, 'dilatedcol.png' ), dilatedcol )

    #識別豎線
    scale = 6
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,rows//scale))
    eroded = cv2.erode(binary,kernel,iterations = 1)
    dilatedrow = cv2.dilate(eroded,kernel,iterations = 1)
    # cv2.imshow("表格豎線展示：",dilatedrow)
    # cv2.waitKey(0)
    # cv2.imwrite( os.path.join( output_folder, 'dilatedrow.png' ), dilatedrow )

    #標識交點
    bitwiseAnd = cv2.bitwise_and(dilatedcol,dilatedrow)
    # cv2.imshow("表格交點展示：",bitwiseAnd)
    # cv2.waitKey(0)
    # cv2.imwrite( os.path.join( output_folder, 'bitwiseAnd.png' ), bitwiseAnd )

    #標識表格
    merge = cv2.add(dilatedcol,dilatedrow)
    # cv2.imshow("表格整體展示：",merge)
    # cv2.waitKey(0)
    # cv2.imwrite( os.path.join( output_folder, 'merge.png' ), merge )


    #兩張圖片進行減法運算，去掉表格框線
    merge2 = cv2.subtract(binary,merge)
    # cv2.imshow("圖片去掉表格框線展示：",merge2)
    # cv2.waitKey(0)
    # cv2.imwrite( os.path.join( output_folder, 'merge2.png' ), merge2 )

    #識別黑白圖中的白色交叉點，將橫縱座標取出
    ys,xs = np.where(bitwiseAnd>0)

    mylisty=[] #縱座標
    mylistx=[] #橫座標

    myys=np.sort(ys)
    myxs=np.sort(xs)

    #通過排序，獲取跳變的x和y的值，說明是交點，否則交點會有好多像素值值相近，我只取相近值的最後一點
    #這個10的跳變不是固定的，根據不同的圖片會有微調，基本上爲單元格表格的高度（y座標跳變）和長度（x座標跳變）
    i = 0
    for i in range(len(myxs)-1):
        if(myxs[i+1]-myxs[i]>10):
            mylistx.append(myxs[i])
        i=i+1
    mylistx.append(myxs[i]) #要將最後一個點加入


    i = 0
    for i in range(len(myys)-1):
        if(myys[i+1]-myys[i]>10):
            mylisty.append(myys[i])
        i=i+1
    mylisty.append(myys[i]) #要將最後一個點加入

    # print('mylisty',mylisty)
    # print('mylistx',mylistx)

    # Destroying present windows on screen
    cv2.destroyAllWindows()

    return mylistx, mylisty

def main( dir_path ):
    # dir_path = opt.path

    details_json = {}
    files_list = next(os.walk(dir_path), (None, None, []))[2]
    img_files = [ img_file for img_file in files_list if not img_file.endswith('.json')] 
    for img_file in img_files:
        img_name = img_file.rsplit('.', 1)[0]
        fcf_map = {}
        img = cv2.imread( os.path.join(dir_path, img_file) )
        gray_img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
        mylistx, mylisty = get_coordinate( img )

        #二值化
        ret, threshold_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

        try:
            symbol_img = threshold_img[ mylisty[0] : mylisty[-1] , mylistx[0] : mylistx[1] ]
            stated_tolerance_img = threshold_img[ mylisty[0] : mylisty[-1] , mylistx[1] : mylistx[2] ]
            primary_datum = None
            secondary_datum = None
            tertiary_datum = None
            if len(mylistx) >= 4:
                primary_datum = threshold_img[ mylisty[0] : mylisty[-1] , mylistx[2] : mylistx[3] ]
            if len(mylistx) >= 5:
                secondary_datum = threshold_img[ mylisty[0] : mylisty[-1] , mylistx[3] : mylistx[4] ]
            if len(mylistx) >= 6:
                tertiary_datum = threshold_img[ mylisty[0] : mylisty[-1] , mylistx[4] : mylistx[5] ]
        except:
            print(f'There is something wrong with the picture {img_file}')
            continue

        symbol_img = image_preprocessing(symbol_img)    
        fcf_map[ 'symbol' ] = Tess_OCR('symbol', symbol_img)


        stated_tolerance_img = image_preprocessing(stated_tolerance_img)    
        fcf_map[ 'stated_tolerance' ] = Tess_OCR('stated_tolerance', stated_tolerance_img)

        if primary_datum is not None:
            primary_datum = image_preprocessing(primary_datum)    
            fcf_map[ 'primary_datum' ] = Tess_OCR('primary_datum', primary_datum)


        if secondary_datum is not None:
            secondary_datum = image_preprocessing(secondary_datum)    
            fcf_map[ 'secondary_datum' ] = Tess_OCR('secondary_datum', secondary_datum)

        
        if tertiary_datum is not None:
            tertiary_datum = image_preprocessing(tertiary_datum)
            fcf_map[ 'tertiary_datum' ] = Tess_OCR('tertiary_datum', tertiary_datum)

        details_json[ img_name ] = fcf_map

    json_path = os.path.join(dir_path, 'recognize.json')
    with open(json_path, 'w', encoding='utf-8') as outfile:
        json.dump(details_json, outfile)

    return


parser = argparse.ArgumentParser()

parser.add_argument('--path', help='FCF image files dir path')

opt = parser.parse_args()

if __name__ == '__main__':
    main( opt.path )
