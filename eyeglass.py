import realtime_glasses_detection.eyeglass_detector as eyeglasses
import dlib
import cv2 as cv
import numpy as np

def has_glasses(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    predictor_path = './realtime_glasses_detection/data/shape_predictor_5_face_landmarks.dat'
    predictor = dlib.shape_predictor(predictor_path)
    height = np.size(img, 0)
    width = np.size(img, 1)
    landmarks = predictor(img, dlib.rectangle(0, 0, width, height))
    landmarks = eyeglasses.landmarks_to_np(landmarks)
    LEFT_EYE_CENTER, RIGHT_EYE_CENTER = eyeglasses.get_centers(img, landmarks)
    aligned_face = eyeglasses.get_aligned_face(gray, LEFT_EYE_CENTER, RIGHT_EYE_CENTER)
    return eyeglasses.judge_eyeglass(aligned_face)

def glasses_map(face_names, face_images):
    return {
        name: has_glasses(img)
        for (name, img) in zip(face_names, face_images)
    }
