import numpy as np
import cv2 as cv

# Coordinates and images come from: https://github.com/kunalgupta777/OpenCV-Face-Filters

def put_moustache(fc,x,y,w,h):
    fc_height = np.size(fc, 0)
    fc_width = np.size(fc, 1)

    face_width = w
    face_height = h

    mst_width = int(face_width*0.4166666)+1
    mst_height = int(face_height*0.142857)+1

    mst = cv.resize(mst_img,(mst_width,mst_height))

    x1, x2 = x + int(0.29166666666*face_width), x + int(0.29166666666*face_width)+mst_width
    y1, y2 = y + int(0.62857142857*face_height), y + int(0.62857142857*face_height)+mst_height

    fc_x1, fc_x2 = max(0, x1), min(fc_width, x2)
    fc_y1, fc_y2 = max(0, y1), min(fc_height, y2)

    mst_x1, mst_x2 = abs(x1 - fc_x1), mst_width - abs(x2 - fc_x2)
    mst_y1, mst_y2 = abs(y1 - fc_y1), mst_height - abs(y2 - fc_y2)

    alpha_s = mst[mst_y1:mst_y2, mst_x1:mst_x2, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        fc_area = fc[fc_y1:fc_y2, fc_x1:fc_x2, c]
        mst_area = mst[mst_y1:mst_y2, mst_x1:mst_x2, c]
        fc[fc_y1:fc_y2, fc_x1:fc_x2, c] = (alpha_s * mst_area + alpha_l * fc_area)

    return fc

def put_hat(fc,x,y,w,h):
    fc_height = np.size(fc, 0)
    fc_width = np.size(fc, 1)

    face_width = w
    face_height = h

    hat_width = face_width * 2
    hat_height = int(0.5*face_height)+1

    hat = cv.resize(hat_img,(hat_width,hat_height))

    x1, x2 = x - face_width // 2, x - face_width // 2 + hat_width
    y1, y2 = y - int(0.5*face_height), y - int(0.5*face_height) + hat_height

    fc_x1, fc_x2 = max(0, x1), min(fc_width, x2)
    fc_y1, fc_y2 = max(0, y1), min(fc_height, y2)

    hat_x1, hat_x2 = abs(x1 - fc_x1), hat_width - abs(x2 - fc_x2)
    hat_y1, hat_y2 = abs(y1 - fc_y1), hat_height - abs(y2 - fc_y2)

    alpha_s = hat[hat_y1:hat_y2, hat_x1:hat_x2, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        fc_area = fc[fc_y1:fc_y2, fc_x1:fc_x2, c]
        hat_area = hat[hat_y1:hat_y2, hat_x1:hat_x2, c]
        fc[fc_y1:fc_y2, fc_x1:fc_x2, c] = (alpha_s * hat_area + alpha_l * fc_area)

    return fc

def put_dog_filter(fc,x,y,w,h):
    fc_height = np.size(fc, 0)
    fc_width = np.size(fc, 1)

    face_width = w
    face_height = h

    dog_width = int(face_width*1.6)
    dog_height = int(face_height*2.8)

    dog = cv.resize(dog_img,(dog_width, dog_height))

    x1, x2 = x - int(0.4*w), x - int(0.4*w) + dog_width
    y1, y2 = y - int(0.8*h) - 1, y - int(0.8*h) - 1 + dog_height

    fc_x1, fc_x2 = max(0, x1), min(fc_width, x2)
    fc_y1, fc_y2 = max(0, y1), min(fc_height, y2)

    dog_x1, dog_x2 = abs(x1 - fc_x1), dog_width - abs(x2 - fc_x2)
    dog_y1, dog_y2 = abs(y1 - fc_y1), dog_height - abs(y2 - fc_y2)

    alpha_s = dog[dog_y1:dog_y2, dog_x1:dog_x2, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        fc_area = fc[fc_y1:fc_y2, fc_x1:fc_x2, c]
        dog_area = dog[dog_y1:dog_y2, dog_x1:dog_x2, c]
        fc[fc_y1:fc_y2, fc_x1:fc_x2, c] = (alpha_s * dog_area + alpha_l * fc_area)

    return fc

mst_img = cv.imread('./filters/moustache.png', cv.IMREAD_UNCHANGED)
hat_img = cv.imread('./filters/cowboy_hat.png', cv.IMREAD_UNCHANGED)
dog_img = cv.imread('./filters/dog_filter.png', cv.IMREAD_UNCHANGED)