import cv2
from ultralytics import YOLO
import numpy as np
import math
import time
import torch
import sys
import myFunctions as myF

# Set GPU for processing
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device: ", device)
if torch.cuda.is_available():
    print("GPU đang được sử dụng.")
else:
    print("Không có GPU nào được sử dụng.")


# Load the YOLOv8 model
machete_model = YOLO("best.pt")
person_model = YOLO("yolov8n.pt")

# Move the models to the desired device
machete_model.model.to(device)
person_model.model.to(device)

# Open the video file
mode = "ap"
if mode == "app":
    if len(sys.argv) > 1:
        parameter = sys.argv[1]
        print(f'Received parameter: {parameter}')
    else:
        print('No parameter received.')

    video_path = 0 if parameter == "stream_camera" else parameter
else:
    video_path = "a.mp4"

cap = cv2.VideoCapture(video_path)
'''if not cap.isOpened():
    print("Failed to open file with FFMPEG. Trying to open with CUDA...")
    cap = cv2.VideoCapture(video_path, cv2.CAP_CUDA)
    if not cap.isOpened():
        print("Failed to open file with CUDA. Exiting...")
        quit()'''

# ----------------------- Initialize parameters: -----------------------
MAX_OBJECTS_TRACKING = 2000  # ----- in def: tracking_object
MAX_HISTORY_TRACKING_OBJECTS = 1000  # ----- in def: tracking_object
THRESHOLD_FOR_DETECT_MACHETE = 0.5  # ----- in def: get_machete_boxes
MIN_DISTANCE_TIME_BETWEEN_WARNING_MACHETE = 3  # ----- in def: check_distance_warning
MIN_DISTANCE_TIME_FOR_TOO_LONG =  30/60 * 60  # ----- in def: check_person_wandering --> check if someone đang ở trong khung hình đủ lâu.
MIN_TOTAL_MOVEMENT_FOR_MOVEMENT = 2 / 1280 * 1280  # ----- in def: check_total_movement --> check if someone is moving những bước đủ lớn.
MIN_DIRECTION_CHANGES_FOR_GOING_BACK_AND_FORTH = 0  # ----- in def: check_going_back_and_forth --> số lần chuyển hướng để được xem là đi qua đi lại.

MODE_FOR_WARNING_MACHETE = ["có người đang mang hung khí", "người đang mang hung khí tiến vào trong khu vực giám sát"]
MODE_FOR_WARNING_MACHETE = MODE_FOR_WARNING_MACHETE[0]
MODE_FOR_DETECT_WANDERING = ["chỉ cần xuất hiện trong khung hình trên 5 phút", \
                             "xuất hiện trong khung hình 5 phút + tổng khoảng cách di chuyển lớn + đi qua đi lại nhiều lần"]
MODE_FOR_DETECT_WANDERING = MODE_FOR_DETECT_WANDERING[1]

# ---------------------------------------------------------------------
FRAME_CAMERA = 24
width, height = 1280, 720
# area = (675, 450, 540, 425)  #----- area = (x_center, y_center, width, height)
line = ((0, int(height * 0.5)),
        (width, int(height * 0.5)))  # ----- line = ((x_start, y_start), (x_end, y_end)) = (start_point, end_point)

# Khởi tạo biến để tính fps
start_time = time.time()
frame_count = 0

# Lưu trữ lịch sử tracking của person và machete
# person_tracking = myF.LimitedDict(max_size=1000)  #----- # Create a LimitedDict with a maximum size of 1000
# machete_tracking = myF.LimitedDict(max_size=1000)
person_tracking, machete_tracking = {}, {}  # ----- object_tracking = {object_id: [[x1,y1,w1,h1], [x2,y2,w2,h2], ...]} --> len(.) = MAX_HISTORY in def: tracking_object
# ----------------------- END OF Initialize parameters -----------------------


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:

        frame_count += 1
        if frame_count % 1 != 0:
            continue

        frame = myF.convert_frame_to_gpu(frame, device)
        frame = myF.calculate_FPS(frame, frame_count, start_time)
        frame = myF.setupFrame(frame, width, height, line)

        warning = False
        exist_machete_brought = False

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        person_results = person_model.track(frame, persist=True)
        machete_results = machete_model.track(frame, persist=True)
        person_tracking, person_current_IDs = myF.tracking_object(person_results, person_tracking,
                                                                  MAX_HISTORY=MAX_HISTORY_TRACKING_OBJECTS,
                                                                  MAX_OBJECTS=MAX_OBJECTS_TRACKING)
        machete_tracking, machete_current_IDs = myF.tracking_object(machete_results, machete_tracking,
                                                                    MAX_HISTORY=MAX_HISTORY_TRACKING_OBJECTS,
                                                                    MAX_OBJECTS=MAX_OBJECTS_TRACKING)

        # Get boxes of person and machete: only get boxes with hight confidence score.
        person_boxes = myF.get_person_boxes(person_results)
        machete_boxes, machete_confidences, machete_IDs_highConf = myF.get_machete_boxes(machete_results,
                                                                                         threshold=THRESHOLD_FOR_DETECT_MACHETE)

        # Kiểm tra người đang mang hung khí:
        macheteBoxes_brought_person = myF.get_macheteBoxes_brought_person(person_boxes, machete_boxes,
                                                                          machete_IDs_highConf)  # ----- macheteBoxes_brought_person = {machete_ID1: [x_cur, y_cur, width_cur, height_cur], machete_ID2: [...], ...} --> macheteBoxes_brought_person contains machete_box if it was being brought by a person, and its corresponding ID.
        if macheteBoxes_brought_person:  # ----- Check if the dictionary is not empty
            exist_machete_brought = True

        # Visualize the detection of machetes and people on the frame:
        annotated_frame = person_results[0].plot()  # ----- show id of people on frame.
        myF.plot_machete_tracking(annotated_frame, machete_boxes, machete_confidences,
                                  machete_IDs_highConf)  # ----- plot high confidence machete boxes and its ID.

        # ---------- Check if người cầm hung khí tiến vào trong khu vực giám sát ----------
        ids_check = list(
            macheteBoxes_brought_person.keys())  # ----- only check for machete_id which brought by a person.
        going_inside = myF.check_going_inside(machete_tracking, ids_check,
                                              line)  # ----- going_inside = {macheteID1: True, macheteID_2: False, ...} of machete_id brought by person if it is going inside.

        # ----- Show WARNING label on screen -----
        warning = myF.making_warning_decision(exist_machete_brought, going_inside,
                                              MIN_DISTANCE_WARNING=MIN_DISTANCE_TIME_BETWEEN_WARNING_MACHETE,
                                              mode=MODE_FOR_WARNING_MACHETE)
        if warning == True:
            annotated_frame = myF.putText(annotated_frame, (300, 300, 400, 200), "WARNING!!!", font_scale=2,
                                          font_thickness=3, label_color=(0, 0, 255))

        # ---------- KIỂM TRA NGƯỜI ĐI LẢNG VẢNG ----------
        wandering_track = myF.check_wandering(person_tracking, person_current_IDs,
                                              MIN_DISTANCE_TIME=MIN_DISTANCE_TIME_FOR_TOO_LONG, \
                                              MIN_TOTAL_MOVEMENT=MIN_TOTAL_MOVEMENT_FOR_MOVEMENT, \
                                              MIN_DIRECTION_CHANGES=MIN_DIRECTION_CHANGES_FOR_GOING_BACK_AND_FORTH, \
                                              mode=MODE_FOR_DETECT_WANDERING)  # ----- wandering_track = {personID_1: True, personID_2: False, ... } check if current people are wandering.

        # Plot wandering warning:
        count_wandering = 0
        for object_id, status in wandering_track.items():
            if status == True:
                annotated_frame = myF.putText(annotated_frame, (225, 700 + count_wandering * 30, 400, 200),
                                              "Person ID: " + str(object_id) + " is wandering..", font_scale=1,
                                              font_thickness=2, label_color=(0, 0, 255))
            count_wandering += 1

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if ESC is pressed: #if cv2.waitKey(1) & 0xFF == ord("q"):
        key = cv2.waitKey(1)
        if key == 27:
            break


    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()