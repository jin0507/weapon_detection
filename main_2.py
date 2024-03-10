import cv2
import numpy as np
import math
import datetime
import pandas as pd
import math
import time
import torch
from collections import OrderedDict


class LimitedDict(OrderedDict):
    def __init__(self, max_size):
        super().__init__()
        self.max_size = max_size

    def __setitem__(self, key, value):
        if len(self) >= self.max_size:
            # If the maximum number of pairs is reached, remove the oldest added pair
            self.popitem(last=False)
        super().__setitem__(key, value)


def convert_frame_to_gpu(frame, device):
    # Convert the OpenCV image from BGR to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the numpy array to a PyTorch tensor and normalize to [0, 1]
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0

    # Add a batch dimension
    frame_tensor = frame_tensor.unsqueeze(0).to(device)

    # Convert tensor to numpy
    frame_numpy = frame_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy() * 255
    frame_numpy = frame_numpy.astype(np.uint8)
    frame = cv2.cvtColor(frame_numpy, cv2.COLOR_RGB2BGR)

    return frame


def calculate_FPS(frame, frame_count, start_time):
    # Tính thời gian đã trôi qua
    elapsed_time = time.time() - start_time

    # Tính fps
    fps = frame_count / elapsed_time

    # Display the FPS on the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)

    return frame


def setupFrame(frame, width, height, line):
    frame = cv2.resize(frame, (width, height))

    start_point, end_point = line[0], line[1]
    frame = drawLine(frame, start_point, end_point)

    return frame


def drawLine(image, start_point, end_point, color=(0, 0, 255), thickness=2):
    cv2.line(image, start_point, end_point, color, thickness)

    return image


def remove_first_item(dictionary):
    first_key = next(iter(dictionary))
    dictionary.pop(first_key)

    return first_key


def tracking_object(yolo_results, object_tracking, MAX_HISTORY=[0, 1000][1], MAX_OBJECTS=2000):
    current_IDs = []  # ----- Get ID of objects in the current frame:

    if yolo_results[0].boxes.is_track is False:
        return object_tracking, current_IDs

    for box in yolo_results[0].boxes:
        obj_id = int(box.id[0]) if box.id[0] is not None else -1
        current_IDs.append(obj_id)
        xywh = [float(coor) for coor in box.xywh[0]]
        if obj_id not in object_tracking:
            object_tracking[obj_id] = [xywh]
        else:
            object_tracking[obj_id].append(xywh)
        if len(object_tracking[obj_id]) > MAX_HISTORY:
            object_tracking[obj_id].pop(0)
        if len(object_tracking) > MAX_OBJECTS:
            remove_first_item(object_tracking)

    return object_tracking, current_IDs


def get_person_boxes(person_results):
    person_boxes = []
    for box in person_results[0].boxes:
        cls = int(box.cls[0])
        if cls == 0:
            xywh = [float(coor) for coor in box.xywh[0]]
            person_boxes.append(xywh)

    return person_boxes


def get_machete_boxes(mechete_results, threshold=0.5):
    machete_boxes = []
    confidences = []
    machete_IDs_highConf = []
    for box in mechete_results[0].boxes:
        conf = float(box.conf[0])
        if conf >= threshold:
            xywh = [float(coor) for coor in box.xywh[0]]
            machete_boxes.append(xywh)
            confidences.append(conf)
            object_id = -1 if box.id is None else int(box.id[0])  # ----- Get ID of box tracked.
            machete_IDs_highConf.append(object_id)

    return machete_boxes, confidences, machete_IDs_highConf


def get_macheteBoxes_overlap_person(person_boxes, machete_boxes, machete_IDs):
    """
    Note: machete_IDs is corresponding to machete_boxes
    """
    macheteBoxes_overlap_person = {}  # ----- macheteBoxes_overlap_person contains machete_box if it overlaps with one of person's boxes, and its corresponding ID.: macheteBoxes_overlap_person = {machete_ID1: [x_cur, y_cur, width_cur, height_cur], machete_ID2: [...], ...}
    for personBox in person_boxes:
        for macheteBox, macheteID in zip(machete_boxes, machete_IDs):
            if is_overlap(personBox, macheteBox):
                macheteBoxes_overlap_person[macheteID] = macheteBox

    return macheteBoxes_overlap_person


def get_macheteBoxes_brought_person(person_boxes, machete_boxes, machete_IDs):
    return get_macheteBoxes_overlap_person(person_boxes, machete_boxes, machete_IDs)


def is_overlap(box1, box2):  # -----box1, box2 = (x_center, y_center, width, height)
    x1_tl, y1_tl, x1_br, y1_br = (
        box1[0] - box1[2] / 2,
        box1[1] - box1[3] / 2,
        box1[0] + box1[2] / 2,
        box1[1] + box1[3] / 2,
    )

    x2_tl, y2_tl, x2_br, y2_br = (
        box2[0] - box2[2] / 2,
        box2[1] - box2[3] / 2,
        box2[0] + box2[2] / 2,
        box2[1] + box2[3] / 2,
    )

    # Kiểm tra xem khoảng x và y có chồng lấn không
    x_overlap = max(0, min(x1_br, x2_br) - max(x1_tl, x2_tl))
    y_overlap = max(0, min(y1_br, y2_br) - max(y1_tl, y2_tl))

    return x_overlap > 0 and y_overlap > 0


def drawBoundingBox(image, xywh, text, color=(0, 255, 0), thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5,
                    font_thickness=1, label_color=(255, 255, 255), showResult=False):
    x_center, y_center, box_width, box_height = xywh
    # Calculate the top-left and bottom-right coordinates of the bounding box
    x1 = int(x_center - box_width / 2)
    y1 = int(y_center - box_height / 2)
    x2 = int(x_center + box_width / 2)
    y2 = int(y_center + box_height / 2)

    # Draw the bounding box on the image
    # color = (0, 255, 0)  # Green color (in BGR format)
    # thickness = 2  # Line thickness
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Add a label to the bounding box
    label = text  # Change this label as needed
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # font_scale = 0.5
    # font_thickness = 1
    # label_color = (255, 255, 255)  # White color (in BGR format)
    cv2.putText(image, label, (x1, y1 - 10), font, font_scale, label_color, font_thickness)

    # Show the image with the bounding box and label
    if showResult == True:
        cv2.imshow("Bounding Box with Label", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image


def plot_machete_tracking(frame, machete_boxes, machete_confidences, machete_IDs):
    for box, conf, machete_id in zip(machete_boxes, machete_confidences, machete_IDs):
        x, y, w, h = [int(coor) for coor in box]
        x_tl, y_tf, x_br, y_br = [int(coor) for coor in xywh2xyxy((x, y, w, h))]
        cv2.rectangle(frame, (x_tl, y_tf), (x_br, y_br), (128, 0, 128), 2)
        machete_id = "None" if machete_id == -1 else machete_id
        label = "#id: " + str(machete_id) + " - long machete " + str(round(conf, 2))
        font_color = (255, 255, 255)  # White text color
        rect_color = (128, 0, 128)  # Purple rectangle color
        write_text_on_background(frame, label, x_tl, y_tf, font_color, rect_color, font_scale=1, font_thickness=2)
        # cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2)
        # Draw the text on the frame with the adjusted font size
        # cv2.putText(frame, label, text_position, font, font_size, color, thickness)


def write_text_on_background(frame, label, x, y, font_color=(255, 255, 255), rect_color=(128, 0, 128), \
                             font_scale=1, font_thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX):
    # Get the size of the text box
    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
    # Create a purple rectangle background
    rect_x1, rect_y1 = x, y - text_height
    rect_x2, rect_y2 = x + text_width, y
    cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), rect_color, cv2.FILLED)
    # Put the label text on top of the rectangle
    cv2.putText(frame, label, (x, y), font, font_scale, font_color, font_thickness)


def check_going_inside(object_tracking, interested_id, line):
    return check_machete_brought_going_inside(object_tracking, interested_id,
                                              line)  # ----- check if hung khí được mang đang tiến vào camera


def check_machete_brought_going_inside(machete_tracking, ids_check, line):
    """
    Functions: check if hung khí được mang đang tiến vào camera.
    """
    going_inside = {}
    for object_id in ids_check:
        going_inside[object_id] = False

        if object_id == -1:
            continue

        coors = machete_tracking[object_id]
        if len(coors) < 2:
            continue
        cur_coor = coors[-1][:2]
        prev_coor = coors[-2][:2]
        if is_point_under_line(cur_coor, line) and is_point_under_line(prev_coor, line):
            if cur_coor[1] - prev_coor[1] > 0:
                going_inside[object_id] = True

    return going_inside


def is_going_inside(object_tracking, interested_id, line):
    return is_machete_brought_going_inside(object_tracking, interested_id, line)


def is_machete_brought_going_inside(machete_tracking, ids_check, line):
    for object_id in ids_check:
        coors = machete_tracking[object_id]
        cur_coor = coors[-1][:2]
        prev_coor = coors[-2][:2]
        if is_point_under_line(cur_coor, line) and is_point_under_line(prev_coor, line):
            if cur_coor[1] - prev_coor[1] > 0:
                return True

    return False


def write_log(dataframe, file_path):
    dataframe.to_csv(file_path, encoding="UTF-8", index=False)


def open_log(file_path):
    df = pd.read_csv(file_path, encoding="UTF-8")
    df = df.astype(str)
    return df


def is_point_under_line(point, line):
    (x_start, y_start), (x_end, y_end) = line
    x_point, y_point = point

    # Calculate the equation of the line in slope-intercept form: y = mx + b
    if x_end - x_start == 0:
        # Handle the case where the line is vertical
        if x_point == x_start:
            return True
        else:
            return False
    else:
        slope = (y_end - y_start) / (x_end - x_start)
        y_intercept = y_start - slope * x_start

        # Calculate the y-coordinate of the point on the line
        calculated_y = slope * x_point + y_intercept

        # Check if the point is under the line
        return calculated_y < y_point


def putText(image, xywh, text, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, font_thickness=1,
            label_color=(255, 255, 255)):
    x_center, y_center, box_width, box_height = xywh
    # Calculate the top-left and bottom-right coordinates of the bounding box
    x1 = int(x_center - box_width / 2)
    y1 = int(y_center - box_height / 2)
    x2 = int(x_center + box_width / 2)
    y2 = int(y_center + box_height / 2)

    cv2.putText(image, text, (x1, y1 - 10), font, font_scale, label_color, font_thickness)

    return image


def xywh2xyxy(box):
    """
    Input: box = (x_center, y_center, width, height) of a box.
    Output: (x_topLeft, y_topLeft, x_rightBottom, y_rightBottom) of that box.
    """
    x_tl, y_tl, x_br, y_br = (
        box[0] - box[2] / 2,
        box[1] - box[3] / 2,
        box[0] + box[2] / 2,
        box[1] + box[3] / 2,
    )

    return (x_tl, y_tl, x_br, y_br)


def check_distance_warning(going_inside, MIN_DISTANCE_WARNING=3):
    distance_warning = False  # ----- return True if there's an object warned: thỏa mãn yêu cầu khoảng cách thời gian tối thiểu.
    log_file_path = "log_files/warning_log.csv"
    df = open_log(log_file_path)

    for object_id, status in going_inside.items():
        if status == True:
            current_time = datetime.datetime.now()
            if str(object_id) not in df['machete_id'].values:
                distance_warning = True
                new_row = {"machete_id": object_id, "time": current_time, "start_time": current_time}
                new_row["warning"] = distance_warning
                new_row_df = pd.DataFrame([new_row])
                df = pd.concat([new_row_df, df], ignore_index=True)
            else:
                index_of_last = df[df['machete_id'] == str(object_id)].index[0]
                prev_time = df.loc[index_of_last, "time"]
                prev_time_object = datetime.datetime.strptime(prev_time, "%Y-%m-%d %H:%M:%S.%f")
                distance_seconds = float((current_time - prev_time_object).seconds)
                if distance_seconds > MIN_DISTANCE_WARNING:
                    distance_warning = True
                    new_row = {"machete_id": object_id, "time": current_time}
                    new_row["warning"] = distance_warning
                    new_row_df = pd.DataFrame([new_row])
                    df = pd.concat([new_row_df, df], ignore_index=True)

    write_log(df, log_file_path)

    return distance_warning


def making_warning_decision(exist_machete_brought, going_inside, MIN_DISTANCE_WARNING, mode):
    warning = False
    print("Checking for: ", mode)

    if mode == "có người đang mang hung khí":
        going_inside_fake = {key: True for key in going_inside}
        warning = check_distance_warning(going_inside_fake, MIN_DISTANCE_WARNING)

    elif mode == "người đang mang hung khí tiến vào trong khu vực giám sát":
        if exist_machete_brought and going_inside:
            warning = check_distance_warning(going_inside, MIN_DISTANCE_WARNING)

    return warning


def check_person_wandering(person_tracking, current_person_IDs, MIN_DISTANCE_TIME=5 * 60):
    """
    MIN_DISTANCE_TIME = 5 * 60  #----- 5 minutes.
    """
    wandering_track = {}  # ----- wandering_track = {personID_1: True, personID_2: False, ... } check if a person is wandering.
    log_file_path = "log_files/person_tracking.csv"
    df = open_log(log_file_path)
    for object_id in current_person_IDs:
        wandering_track[object_id] = False
        coords = person_tracking[object_id]
        current_time = datetime.datetime.now()
        new_row = {"person_id": object_id, "time": current_time}
        if str(object_id) not in df['person_id'].values:
            new_row["start_time"] = current_time
        else:
            index_of_first = df[df['person_id'] == str(object_id)].index[-1]
            start_time = df.loc[index_of_first, "start_time"]
            start_time_object = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S.%f")
            distance_seconds = float((current_time - start_time_object).seconds)
            if distance_seconds >= MIN_DISTANCE_TIME:
                wandering_track[object_id] = True

        new_row["wandering"] = wandering_track[object_id]

        new_row_df = pd.DataFrame([new_row])
        df = pd.concat([new_row_df, df], ignore_index=True)

    write_log(df, log_file_path)

    return wandering_track


def check_wandering(object_tracking, current_IDs, MIN_DISTANCE_TIME, MIN_TOTAL_MOVEMENT, MIN_DIRECTION_CHANGES, mode):
    print("Checking for LẢNG VẢNG: ", mode)

    if mode == "chỉ cần xuất hiện trong khung hình trên 5 phút":
        return check_person_wandering(person_tracking=object_tracking, current_person_IDs=current_IDs,
                                      MIN_DISTANCE_TIME=MIN_DISTANCE_TIME)

    elif mode == "xuất hiện trong khung hình 5 phút + tổng khoảng cách di chuyển lớn + đi qua đi lại nhiều lần":
        return check_person_wandering_v2(person_tracking=object_tracking, current_person_IDs=current_IDs, \
                                         MIN_DISTANCE_TIME=MIN_DISTANCE_TIME, MIN_TOTAL_MOVEMENT=MIN_TOTAL_MOVEMENT, \
                                         MIN_DIRECTION_CHANGES=MIN_DIRECTION_CHANGES)


# --------------------------------- CHECK WANDERING KẾT HỢP 3 YẾU TỐ: di chuyển, ở lâu, đi qua đi lại. -----------------------------------------
def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    euclidean_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return euclidean_distance


def check_total_movement(object_id, tracking_object, MIN_TOTAL_MOVEMENT=2 * 1280):
    coors = tracking_object[object_id]
    total_distance_movement = sum([distance(coors[i][:2], coors[i - 1][:2]) for i in range(1, len(coors))])

    return total_distance_movement > MIN_TOTAL_MOVEMENT, total_distance_movement


def check_person_too_long(person_id, log_file_path="log_files/person_tracking.csv", MIN_DISTANCE_TIME=5 * 60):
    df = open_log(log_file_path)
    too_long = False

    current_time = datetime.datetime.now()
    new_row = {"person_id": person_id, "time": current_time}
    if str(person_id) not in df["person_id"].values:
        new_row["start_time"] = current_time
    else:
        index_of_first = df[df['person_id'] == str(person_id)].index[-1]
        start_time = df.loc[index_of_first, "start_time"]
        start_time_object = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S.%f")
        distance_seconds = float((current_time - start_time_object).seconds)
        if distance_seconds >= MIN_DISTANCE_TIME:
            too_long = True
            new_row["too_long"] = True

    new_row_df = pd.DataFrame([new_row])
    df = pd.concat([new_row_df, df], ignore_index=True)

    write_log(df, log_file_path)

    return too_long


def compute_horizontal_direction_changes(coords):
    direction_changes = 0
    prev_dx = None

    for i in range(1, len(coords)):
        dx = coords[i][0] - coords[i - 1][0]

        # Kiểm tra xem dx có thay đổi dấu so với dx trước đó hay không
        if prev_dx is not None and dx * prev_dx < 0:
            direction_changes += 1

        prev_dx = dx

    return direction_changes


def check_going_back_and_forth(num_direction_changes, MIN_DIRECTION_CHANGES=2):
    return num_direction_changes > MIN_DIRECTION_CHANGES


def check_person_wandering_v2(person_tracking, current_person_IDs, MIN_DISTANCE_TIME, MIN_TOTAL_MOVEMENT,
                              MIN_DIRECTION_CHANGES):
    """
    Note: Kết hợp 3 yếu tố: di chuyển, ở lâu, đi qua đi lại.
    """

    wandering_track = {}

    for person_ID in current_person_IDs:
        wandering_track[person_ID] = False

        is_movement, total_distance_movement = check_total_movement(person_ID, person_tracking,
                                                                    MIN_TOTAL_MOVEMENT=MIN_TOTAL_MOVEMENT)
        too_long = check_person_too_long(person_ID, log_file_path="log_files/person_tracking.csv",
                                         MIN_DISTANCE_TIME=MIN_DISTANCE_TIME)

        coords = person_tracking[person_ID]
        num_direction_changes = compute_horizontal_direction_changes(coords)
        is_going_back_and_forth = check_going_back_and_forth(num_direction_changes,
                                                             MIN_DIRECTION_CHANGES=MIN_DIRECTION_CHANGES)

        if is_movement and too_long and is_going_back_and_forth:
            wandering_track[person_ID] = True

    return wandering_track