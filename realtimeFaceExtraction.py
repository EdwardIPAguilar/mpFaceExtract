import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# create FaceMesh obj
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

desired_w = 120
desired_h = 120

video_input = cv2.VideoCapture(0)
while video_input.isOpened():
    success, img = video_input.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Apply face mesh
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            coord_indices = set()
            for line in mp_face_mesh.FACEMESH_LIPS:
                coord_indices.add(line[0])
                coord_indices.add(line[1])

            coords = []
            for index in coord_indices:
                lm = face_landmarks.landmark[index]
                coords.append((int(lm.x * img.shape[1]), int(lm.y * img.shape[0])))
            
            if coords:  # check if coords is not empty
                min_x = min(coords, key = lambda t: t[0])[0]
                max_x = max(coords, key = lambda t: t[0])[0]
                min_y = min(coords, key = lambda t: t[1])[1]
                max_y = max(coords, key = lambda t: t[1])[1]

                cur_w = max_x - min_x
                cur_h = max_y - min_y

                center_x = min_x + (cur_w / 2)
                center_y = min_y + (cur_h / 2)
                
                desired_aspect = desired_w / desired_h
                cur_aspect = cur_w / cur_h
                if cur_aspect < desired_aspect:
                  cur_w = cur_h
                else:
                  cur_h = cur_w

                min_x = int(center_x - cur_w / 2)
                max_x = int(center_x + cur_w / 2)
                min_y = int(center_y - cur_w / 2)
                max_y = int(center_y + cur_w / 2)

                img_w = img.shape[1]
                img_h = img.shape[0]
                #in the case that expanding the bounding box exceeds original image size, add padding
                oob_left = 0
                oob_right = 0
                oob_top = 0
                oob_bottom = 0
                if min_x < 0:
                    oob_left = -min_x
                    min_x = 0
                if max_x > img_w:
                    oob_right = max_x - img_w 
                    max_x = img_w - 1
                if min_y < 0:
                    oob_top = -min_y
                    min_y = 0
                if max_y > img_h:
                    oob_bottom = max_y - img_h
                    max_y = img_h - 1  

                # Crop & resize the image
                img = img[min_y:max_y, min_x:max_x]
                img = cv2.copyMakeBorder(img, oob_top, oob_bottom, oob_left, oob_right,
                                         cv2.BORDER_CONSTANT, value=[0, 0, 0])
                img = cv2.resize(img, (500, 500))

            else:
                print('coords is empty, likely an issue in returning landmarks')

    cv2.imshow('EchoLabs_Window', img)
    if cv2.waitKey(20) & 0xFF == ord("q"): #exit camera by pressing 'q'
        break

video_input.release()
cv2.destroyAllWindows()