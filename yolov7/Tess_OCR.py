import cv2
import numpy as np
import pytesseract


def getProjection(image):

    (h, w) = image.shape

    start = 0
    end = 0
    isFirst = True

    for i in range(h):
        arr = image[i]
        re_arr = np.flipud(arr)

        if ~(np.any(arr)):
            continue

        if isFirst:
            start = i - 1
            isFirst = False
        else:
            end = i + 1

    return start, end

def image_preprocessing( original_image ):

    # cv2.imshow('Original Image', original_image)
    # cv2.waitKey(0)

    h, w = original_image.shape[:2]

    side_length = int(max( h, w )*1.3)
    blank_image = np.zeros( (side_length, side_length, 3), np.uint8 )
    blank_image[:,:] = (255,255,255)
    border_y, border_x = int((side_length-h)/2), int((side_length-w)/2)
    blank_image[ border_y : border_y + h, border_x : border_x + w ] = original_image
    image = blank_image.copy()
    # cv2.imshow('New Image', image)
    # cv2.waitKey(0)


    # 轉化成灰度圖
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # flip the foreground and background to ensure text is "white"
    image_gray = cv2.bitwise_not(image_gray)
    image_blur = cv2.GaussianBlur(image_gray, (7,7), 0)

    # setting all foreground pixels to 255 and all background pixels to 0
    ret, thresh = cv2.threshold(image_blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    whereid = np.where(thresh > 0)
    # 交換橫縱坐標的順序，否則下面得到的每個像素點為(y,x)
    whereid = whereid[::-1]
    # 將像素點格式轉換為(n_coords, 2)，每個點表示為(x,y)
    coords = np.column_stack(whereid)

    (x,y), (w,h), angle = cv2.minAreaRect(coords)
    # print('[angle]', angle)
    if angle < -45:
        angle = 90 - angle
    # angle = 90 - angle

    vis = image.copy()
    box = cv2.boxPoints(((x,y), (w,h), angle))
    box = np.int0(box)
    cv2.drawContours(vis,[box],0,(0,0,255),2)

    # rotate the image to deskew it
    center = (side_length // 2, side_length // 2)
    # center = x, y  # 可以试试中心点设置为文本区域中心会是什么情况
    Mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(image, Mat, (side_length, side_length), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    image_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    ret, image_binary = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('Image Gray', thresh)
    # cv2.waitKey(0)

    H_start, H_end = getProjection(image_binary)
    W_start, W_end = getProjection(image_binary.T)

    border = round(side_length*0.05)
    image_crop = rotated[ H_start - border : H_end + border, W_start - border : W_end + border ]


    (h, w, d) = image_crop.shape
    num = 10
    pattern_image = np.zeros( (h*num, w*num, 3), np.uint8 )

    for i in range(num):
        for j in range(num):
            pattern_image[ h*i: h*(i+1), w*j: w*(j+1) ] = image_crop

    rotate_data = pytesseract.image_to_osd(pattern_image, output_type=pytesseract.Output.DICT)
    rotate = float(rotate_data["rotate"])
    # print("[ANGLE] "+str(rotate))

    image_rotated = np.rot90(image_crop, -rotate/90)
    # cv2.imshow('Image Crop', image_crop)
    # cv2.imshow('Image Rotated', image_rotated)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return image_rotated


def recg_text( original_image, lang='eng', config="--oem 3 --psm 4" ):

    datas = pytesseract.image_to_data(original_image, lang=lang, config=config,output_type=pytesseract.Output.DICT)
    # print("[DATA]"+str(datas))

    result = []
    for i in range(len(datas["text"])):
        if datas["conf"][i] > 10:
            res = {
                'x': datas['left'][i],
                'y': datas['top'][i],
                'w': datas['width'][i],
                'h': datas['height'][i],
                'text': datas['text'][i]
            }
            result.append(res)
            (x, y, w, h) = (datas['left'][i], datas['top'][i], datas['width'][i], datas['height'][i])
            cv2.rectangle(original_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # cv2.imshow('Image', original_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return result

if __name__ == '__main__':
    img_path = "DIM_02.jpg"
    img = cv2.imread(img_path)

    image_rotated = image_preprocessing( img )
    
    result = recg_text( image_rotated )

    