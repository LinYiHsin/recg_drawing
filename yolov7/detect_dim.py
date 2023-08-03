import argparse
import time
import json
from pathlib import Path
import pytesseract

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image, ImageDraw, ImageFont

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.torch_utils import select_device
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

def rotate_img( image ):

    h, w = image.shape[:2]

    side_length = int(max( h, w )*1.3)

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

    if angle > 45:
        angle = angle - 90
    # angle = 90 - angle
    # print('[angle]', angle)

    vis = image.copy()
    box = cv2.boxPoints(((x,y), (w,h), angle))
    box = np.int0(box)
    cv2.drawContours(vis,[box],0,(0,0,255),2)

    # rotate the image to deskew it
    center = (side_length // 2, side_length // 2)
    # center = x, y  # 可以试试中心点设置为文本区域中心会是什么情况
    Mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(image, Mat, (side_length, side_length), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


    return rotated

def image_preprocessing( image ):

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((2,2), np.uint8)

    dilate_img = cv2.dilate(gray_img, kernel, iterations = 1)

    erode_img = cv2.erode(dilate_img, kernel, iterations = 1)

    return erode_img

def recognize(image, image_name, det_map, names):
    datas = []
    for index, item in enumerate(det_map):
        xyxy = item['xyxy']
        cls = item['class']
        label = names[int(cls)]

        x1, y1, x2, y2 = xyxy
        w, h = abs(x2-x1), abs(y2-y1)
        side_length = int( max( w, h ) * 1.2 )
        new_image = np.zeros( (side_length, side_length, 3), np.uint8 )
        new_image[:,:] = (255,255,255)
        y_offset, x_offset = int((side_length-h)/2), int((side_length-w)/2)

        img = image.copy()
        img = img[int(y1): int(y2), int(x1): int(x2)]
        new_image[ int(y_offset) : int(y_offset+h), int(x_offset) : int(x_offset+w) ] = img

        new_image = image_preprocessing(new_image)
        # rotated_image = rotate_img(new_image)
        # cv2.imshow('Image rotated', rotated_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        output = pytesseract.image_to_data(new_image, config="-l eng_math --psm 6 --oem 3",output_type=pytesseract.Output.DICT)
        text_list = [x for x in output['text'] if x != '' ]
        data = {'label': label, 'xyxy': [int(x1), int(y1), int(x2), int(y2)], 'text': ''.join(text_list)}
        datas.append(data)

    # print("[DATA]"+str(datas))
    details = {'image_name': image_name, 'datas': datas}
    # with open(save_path, 'w', encoding='utf-8') as outfile:
    #     json.dump(details, outfile)

    return details

def plot_label_box(x, img, color=None, label=None, conf=None):
    # Plots one bounding box on image img
    tl = round(0.002 * max(img.shape[0], img.shape[1]))
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        # t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, color, thickness=tf, lineType=cv2.LINE_AA)

    return


def detect(save_img=True):
    source, weights, view_img, save_txt, img_size, trace, rotate = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace, opt.rotate_result
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    img_size = check_img_size(img_size, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=img_size, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = img_size
    old_img_b = 1

    json_path = str(save_dir / 'recognize.json')
    details_json = []

    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=img_size, stride=stride)
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            im0_cpoy = im0.copy()

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                det_map = [ {'xyxy': xyxy, 'conf': conf, 'class': cls} for *xyxy, conf, cls in det ]
                details_json.append(recognize(im0_cpoy, p.name, det_map, names))

                det_map = sorted(det_map, key=lambda d: d['xyxy'][0]) 
                for index, item in enumerate(det_map):
                    xyxy = item['xyxy']
                    conf = item['conf']
                    cls = item['class']

                    # Add bbox to image
                    plot_label_box(xyxy, im0, label=f'{names[int(cls)]}_{index:02d}', conf=conf, color=colors[int(cls)])

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')


            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
        
        with open(json_path, 'w', encoding='utf-8') as outfile:
            json.dump(details_json, outfile)

    return


if __name__ == '__main__':
    t0 = time.time()
    s = time.gmtime(t0)
    time_str = time.strftime("%Y-%m-%d-%H%M%S", s)

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default=time_str, help='save results to project/name')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')

    parser.add_argument('--rotate-result', action='store_true', help='rotate label image')

    opt = parser.parse_args()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()

    print(f'Done. ({time.time() - t0:.3f}s)')
