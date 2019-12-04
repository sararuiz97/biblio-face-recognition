#!/usr/bin/env python3

import load_face
import face_recognition
import cv2 as cv
import numpy as np
from datetime import datetime
import eyeglass
import filters.filters as filters

def delete_old_unknown():
    i = 0
    while i < len(unknown_faces_ids):
        if (datetime.now() - unknown_faces_lastmatch[i]).seconds > 5:
            del unknown_faces_ids[i]
            del unknown_faces_encodings[i]
            del unknown_faces_ages[i]
            del unknown_faces_areas[i]
            del unknown_faces_pixels[i]
            del unknown_faces_lastmatch[i]
            i -= 1
        i += 1

facedir = 'face/'
celebritydir = 'celebrities/'
known_face_names, known_face_encodings, known_face_images = load_face.load_faces(facedir)
eyeglass_map = eyeglass.glasses_map(known_face_names, known_face_images)

celebrity_face_names, celebrity_face_encodings, celebrity_face_images = load_face.load_faces(celebritydir)

# video_capture = cv.VideoCapture('http://192.168.1.68:8080/video')
video_capture = cv.VideoCapture(0)
process_this_frame = 0
resize_factor = 1
unknown_next = 0
unknown_faces_ids = []
unknown_faces_encodings = []
unknown_faces_ages = []
unknown_faces_lastmatch = []
unknown_faces_areas = []
unknown_faces_pixels = []
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv.resize(frame, (0, 0), fx=(1 / resize_factor), fy=(1 / resize_factor))

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame == 0:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        face_changes = []
        lookalike_names = []
        face_is_far = False
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = 'Unknown ' + str(unknown_next)
            found = False
            face_area = (bottom - top) * (right - left)
            has_glasses = eyeglass.has_glasses(small_frame[top:bottom, left:right])

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                found = True

            if not found and len(unknown_faces_encodings) > 0:
                u_matches = face_recognition.compare_faces(unknown_faces_encodings, face_encoding)
                u_face_distances = face_recognition.face_distance(unknown_faces_encodings, face_encoding)
                u_best_match_index = np.argmin(u_face_distances)
                if u_matches[u_best_match_index]:
                    unknown_faces_lastmatch[u_best_match_index] = datetime.now()
                    name = 'Unknown ' + str(unknown_faces_ids[u_best_match_index])
                    # Consider the bigger encoding as better and keep it
                    if face_area > unknown_faces_areas[u_best_match_index]:
                        unknown_faces_areas[u_best_match_index] = face_area
                        unknown_faces_encodings[u_best_match_index] = face_encoding

                        top *= resize_factor
                        right *= resize_factor
                        bottom *= resize_factor
                        left *= resize_factor

                        unknown_faces_pixels[u_best_match_index] = frame[top:bottom, left:right].copy()
                    if unknown_faces_areas[u_best_match_index] > 70000 and (datetime.now() - unknown_faces_ages[u_best_match_index]).seconds > 7:
                        cv.imwrite(facedir + 'Person ' + str(unknown_faces_ids[u_best_match_index]) + '.jpg', unknown_faces_pixels[u_best_match_index])
                        del unknown_faces_ids[u_best_match_index]
                        del unknown_faces_encodings[u_best_match_index]
                        del unknown_faces_ages[u_best_match_index]
                        del unknown_faces_areas[u_best_match_index]
                        del unknown_faces_pixels[u_best_match_index]
                        del unknown_faces_lastmatch[u_best_match_index]
                        known_face_names, known_face_encodings, known_face_images = load_face.load_faces(facedir)
                        eyeglass_map = eyeglass.glasses_map(known_face_names, known_face_images)
                    else:
                        face_is_far = True
                    found = True

            if not found:
                unknown_faces_ids.append(unknown_next)
                unknown_faces_ages.append(datetime.now())
                unknown_faces_lastmatch.append(datetime.now())
                unknown_faces_encodings.append(face_encoding)
                unknown_faces_areas.append(face_area)

                top *= resize_factor
                right *= resize_factor
                bottom *= resize_factor
                left *= resize_factor

                unknown_faces_pixels.append(frame[top:bottom, left:right].copy())

                unknown_next += 1

            glass_change = (name in eyeglass_map) and has_glasses != eyeglass_map[name]
            beard_change = False

            lookalike_distances = face_recognition.face_distance(celebrity_face_encodings, face_encoding)
            lookalike_best_match_index = np.argmin(lookalike_distances)
            lookalike_names.append(celebrity_face_names[lookalike_best_match_index])

            face_names.append(name)
            face_changes.append((glass_change, beard_change))


    delete_old_unknown()

    process_this_frame = (process_this_frame + 1) % 5


    # Display the results
    for (top, right, bottom, left), name, (glass_change, beard_change), lookalike_name in zip(face_locations, face_names, face_changes, lookalike_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= resize_factor
        right *= resize_factor
        bottom *= resize_factor
        left *= resize_factor

        # # Draw a box around the face
        # cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Apply filter
        # filters.put_hat(frame, left, top, right - left, bottom - top)
        # filters.put_moustache(frame, left, top, right - left, bottom - top)

        filters.put_dog_filter(frame, left, top, right - left, bottom - top)

        # Draw a label with a name below the face
        cv.rectangle(frame, (left, bottom - 20), (right, bottom), (0, 0, 255), cv.FILLED)
        font = cv.FONT_HERSHEY_DUPLEX
        cv.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        cv.rectangle(frame, (left, bottom), (right, bottom + 20), (120, 0, 120), cv.FILLED)
        font = cv.FONT_HERSHEY_DUPLEX
        cv.putText(frame, 'Lookalike: ' + lookalike_name, (left + 6, bottom + 20 - 6), font, 0.5, (255, 255, 255), 1)

        if glass_change:
            cv.rectangle(frame, (left, bottom + 20), (right, bottom + 40), (255, 0, 0), cv.FILLED)
            font = cv.FONT_HERSHEY_DUPLEX
            cv.putText(frame, 'What\'s with the glasses?', (left + 6, bottom + 40 - 6), font, 0.5, (255, 255, 255), 1)

    height = np.size(frame, 0)
    width = np.size(frame, 1)
    cv.rectangle(frame, (0, 0), (width, 30), (255, 255, 255), cv.FILLED)
    font = cv.FONT_HERSHEY_DUPLEX
    cv.putText(frame, 'Privacy policy can be accessed at: http://jeje.com', (6, 30 - 6), font, 0.75, (0, 0, 0), 1)

    if face_is_far:
        cv.rectangle(frame, (0, 30), (width, 60), (255, 255, 255), cv.FILLED)
        font = cv.FONT_HERSHEY_DUPLEX
        cv.putText(frame, 'Get closer if you want me to remember your face!', (6, 60 - 6), font, 0.75, (0, 0, 0), 1)

    # Display the resulting image
    cv.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv.destroyAllWindows()
