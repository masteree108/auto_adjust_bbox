# USAGE
# python MOTF_process_pool.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --video race.mp4

# watch CPU loading on the ubuntu
# ps axu | grep [M]OTF_process_pool.py | awk '{print $2}' | xargs -n1 -I{} ps -o sid= -p {} | xargs -n1 -I{} ps --forest -o user,pid,ppid,cpuid,%cpu,%mem,stat,start,time,command -g {}

# import the necessary packages
from imutils.video import FPS
from multiprocessing import Pool
import numpy as np
import argparse
import imutils
import cv2
import os
import time
import mot_class as mtc


def get_algorithm_tracker(algorithm):
    if algorithm == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    elif algorithm == 'MIL':
        tracker = cv2.TrackerMIL_create()
    elif algorithm == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif algorithm == 'TLD':
        tracker = cv2.TrackerTLD_create()
    elif algorithm == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    elif algorithm == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    elif algorithm == 'CSRT':
        tracker = cv2.TrackerCSRT_create()
    elif algorithm == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    return tracker


def read_user_input_info():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True,
                    help="path to input video file")
    ap.add_argument("-o", "--output", type=str,
                    help="path to optional output video file")
    ap.add_argument("-c", "--confidence", type=float, default=0.2,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-u", "--use-gpu", type=int, default=0,
                    help="boolean indicating if CUDA GPU should be used")
    ap.add_argument("-y", "--yolo", required=True,
                    help="base path to YOLO directory")
    ap.add_argument("-t", "--threshold", type=float, default=0.3,
                    help="threshold when applyong non-maxima suppression")
    args = vars(ap.parse_args())

    return args


def init_tracker(bboxes, frame):
    # print("detect_people_,number:%d" % detect_people_num)
    # grab the appropiate object tracker using our dictionary of
    # OpenCV object tracker objects
    # it should brings (left, top, width, height) to tracker.init() function
    # parameters are left, top , right and bottom in the box
    # so those parameters need to minus like below to get width and height
    for i, org_bbox in enumerate(bboxes):
        newbbox = []
        newbbox.append(int(org_bbox[0]))
        newbbox.append(int(org_bbox[1]))
        newbbox.append(int(org_bbox[2]))
        newbbox.append(int(org_bbox[3]))

        bbox = (newbbox[0], newbbox[1], newbbox[2], newbbox[3])
        _multi_tracker.add(get_algorithm_tracker("CSRT"), frame, bbox)


def out_of_bbox_range_frame_paint_black(bboxes, frame):
    offset_x = 0
    offset_y = 0
    (vh, vw) = _frame.shape[:2]
    frame_only_one_person = []
    for i, bbox in enumerate(bboxes):
        x = int(bbox[0])
        y = int(bbox[1])
        w = int(bbox[2])
        h = int(bbox[3])

        offest_x = int(w / 5)
        offest_y = int(h / 5)

        xl = x - offset_x
        if xl < 0:
            xl = 0

        xr = x + w + offset_x
        if xr > vw:
            xr = vw

        yt = y - offset_y
        if yt < 0:
            yt = 0

        yb = y + h + offset_y
        if yb > vh:
            yb = vh

        crop_img = frame[yt:yb, xl:xr]
        img_CMB = cv2.copyMakeBorder(crop_img, yt, vh - yb, xl, vw - xr, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        frame_only_one_person.append(img_CMB)
        # cv2.imwrite("user_pick_" + str(i) + "_" + str(time.time())+"_.png", img_CMB)

    return frame_only_one_person


def crop_people_method(bboxes, frame):
    offset_x = 0
    offset_y = 0
    (vh, vw) = _frame.shape[:2]
    crop_people = []
    for i, bbox in enumerate(bboxes):
        x = int(bbox[0])
        y = int(bbox[1])
        w = int(bbox[2])
        h = int(bbox[3])

        offest_x = int(w / 5)
        offest_y = int(h / 5)

        xl = x - offset_x
        if xl < 0:
            xl = 0

        xr = x + w + offset_x
        if xr > vw:
            xr = vw

        yt = y - offset_y
        if yt < 0:
            yt = 0

        yb = y + h + offset_y
        if yb > vh:
            yb = vh

        crop_img = frame[yt:yb, xl:xr]
        crop_people.append(crop_img)
        # cv2.imwrite("crop_" + str(i) + "_" + str(time.time())+"_.png", crop_img)

    return crop_people


def detect_people_by_ROI(frame):
    ROI_window_name = "multi ROIs(draw bbox ok:Enter or Space, exit:Esc)"
    cv2.namedWindow(ROI_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ROI_window_name, _frame_size_width, _frame_size_height)
    # bbox:posX ,posY, width, length
    bboxes = cv2.selectROIs(ROI_window_name, frame, False)
    cv2.destroyWindow(ROI_window_name)
    return bboxes


def IOU_check(src_bbox, adjust_bboxes):
    print("IoU method")
    iou_temp = []
    boxSRCArea = (src_bbox[2] + 1) * (src_bbox[3] + 1)
    for i, adjust_bbox in enumerate(adjust_bboxes):
        xA = max(src_bbox[0], adjust_bbox[0])
        # print("xA:%.2f" % xA)
        yA = max(src_bbox[1], adjust_bbox[1])
        # print("yA:%.2f" % yA)
        xB = min(src_bbox[0] + src_bbox[2], adjust_bbox[0] + adjust_bbox[2]) - xA
        # print("xB:%.2f" % xB)
        yB = min(src_bbox[1] + src_bbox[3], adjust_bbox[1] + adjust_bbox[3]) - yA
        # print("yB:%.2f" % yB)

        # set max ioy range to
        xB = max(xB, 0)
        yB = max(yB, 0)

        interArea = xB * yB
        # print("interArea:%.2f" % interArea)

        # boxSRCArea = (src_bbox[2] + 1) * (src_bbox[3] + 1)
        boxADJArea = (adjust_bbox[2] + 1) * (adjust_bbox[3] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction  ground-truth
        # areas - the intersection area
        iou = interArea / float(boxSRCArea + boxADJArea - interArea)
        iou_temp.append(iou)

    # if max(iou_temp) > 0:
    iou_array = np.array(iou_temp)
    index = np.argmax(iou_array)
    # count time IOU_check
    try:
        IOU_check.count += 1
    except AttributeError:
        IOU_check.count = 1

    # If it's the first IOU regardless of the IOU score
    if IOU_check.count != 1:
        if iou_array[index] > 0.5:
            print("adjust_bboxes")
            return adjust_bboxes[index]
        else:
            print("src_box")
            return src_bbox
    else:
        return adjust_bboxes[index]


def IOU_check_for_first_frame(src_bbox, adjust_bboxes):
    print("IoU method(first_frame)")
    iou_temp = []
    boxSRCArea = (src_bbox[2] + 1) * (src_bbox[3] + 1)
    for i, adjust_bbox in enumerate(adjust_bboxes):
        xA = max(src_bbox[0], adjust_bbox[0])
        # print("xA:%.2f" % xA)
        yA = max(src_bbox[1], adjust_bbox[1])
        # print("yA:%.2f" % yA)
        xB = min(src_bbox[0] + src_bbox[2], adjust_bbox[0] + adjust_bbox[2])
        # print("xB:%.2f" % xB)
        yB = min(src_bbox[1] + src_bbox[3], adjust_bbox[1] + adjust_bbox[3])
        # print("yB:%.2f" % yB)

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # print("interArea:%.2f" % interArea)

        boxADJArea = (adjust_bbox[2] + 1) * (adjust_bbox[3] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction  ground-truth
        # areas - the intersection area
        iou = interArea / float(boxSRCArea + boxADJArea - interArea)
        iou_temp.append(iou)

    # if max(iou_temp) > 0:
    iou_array = np.array(iou_temp)
    index = np.argmax(iou_array)
    print("iou_array:")
    print(iou_array)

    if iou_array[index] > 0.1:
        return adjust_bboxes[index]
    else:
        return src_bbox


def detect_people_and_get_adjust_bboxes_for_first_frame_paint_black(frame, user_draw_bboxes, frame_only_one_person):
    final_bboxes = []
    get_bboxes = []

    for i, person in enumerate(frame_only_one_person):
        get_bboxes.append([])
        (h, w) = person.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        _net.setInput(blob)
        detections = _net.forward()
        recog_person = True
        layerOutputs = _net.forward(_ln)

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []
        idxs = []
        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > _args["confidence"]:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height

                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    print("A")
                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    if _LABELS[classID] != "person":
                        continue
                    else:
                        # update our list of bounding box coordinates,
                        # confidences, and class IDs
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, _args["confidence"], _args["threshold"])
        print("idxs")
        print(idxs)
        if recog_person == False:
            final_bboxes.append(user_draw_bboxes[i])
        else:
            # IoU check which box is best
            final_bboxes.append(IOU_check_for_first_frame(user_draw_bboxes[i], boxes))

    return final_bboxes


def detect_people_and_get_adjust_bboxes_for_first_frame_all_frame(frame, user_draw_bboxes):
    final_bboxes = []
    get_bboxes = []
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    _net.setInput(blob)
    detections = _net.forward()
    recog_person = False
    layerOutputs = _net.forward(_ln)

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    idxs = []
    print("1")
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > _args["confidence"]:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height

                box = detection[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                print("A")
                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                if _LABELS[classID] != "person":
                    continue
                # update our list of bounding box coordinates,
                # confidences, and class IDs
                else:
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, _args["confidence"], _args["threshold"])
    # debug:watch detect result
    save_result = False
    sframe = frame.copy()
    if save_result == True:
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(sframe, p1, p2, (0, 0, 255), 2)
            (startX, startY) = p1
            cv2.putText(sframe, "model", (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.imwrite("detect_result" + "_" + str(time.time()) + "_.png", sframe)

    # 2. using IoU to find which are match user drawed bboxes
    for i, bbox in enumerate(user_draw_bboxes):
        # IoU check which box is best
        final_bboxes.append(IOU_check_for_first_frame(bbox, boxes))

    return final_bboxes


"""def detect_people_and_get_adjust_bboxes_for_first_frame_crop_people(frame, user_draw_bboxes, crop_people):
    final_bboxes = []
    get_bboxes = []

    for i, person in enumerate(crop_people):
        get_bboxes.append([])
        (h, w) = person.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        _net.setInput(blob)
        layerOutputs = _net.forward(_ln)

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []
        print("1")
        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > _args["confidence"]:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height

                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        if recog_person == False:
            final_bboxes.append(user_draw_bboxes[i])
        else:
            # IoU check which box is best
            final_bboxes.append(IOU_check_for_first_frame(user_draw_bboxes[i], get_bboxes[i]))

    return final_bboxes
"""


def detect_people_and_get_adjust_bboxes2(bboxes, frame, crop_people):
    final_bboxes = []
    get_bboxes = []

    for i, crop_person in enumerate(crop_people):
        ct = i - 1
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        _net.setInput(blob)
        detections = _net.forward()
        recog_person = False
        layerOutputs = _net.forward(_ln)

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []
        idxs = []
        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > _args["confidence"]:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height

                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    if _LABELS[classID] != "person":
                        continue
                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    else:
                        recog_person = True
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, _args["confidence"], _args["threshold"])
        if recog_person == False:
            # final_bboxes.append((0 ,0 ,0 ,0))
            final_bboxes.append(bboxes[ct])
        else:
            # IoU check which box is best
            final_bboxes.append(IOU_check(bboxes[ct], boxes))
    return final_bboxes


def frame_add_w_and_h(frame):
    (vh, vw) = frame.shape[:2]
    cmb_w = 0
    cmb_h = 0
    do_border_w = True
    do_border_h = True
    if vw < _frame_size_width:
        cmb_w = int(abs(_frame_size_width - vw) / 2)
        do_border_w = False

    if vh < _frame_size_height:
        cmb_h = int(abs(_frame_size_height - vh) / 2)
        do_border_h = False

    if do_border_w == True and do_border_h == True:
        img_CMB = cv2.copyMakeBorder(frame, cmb_h, cmb_h, cmb_w, cmb_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return img_CMB
    else:
        return frame


def main():
    # loop over frames from the video file stream
    while True:
        try:
            main.count += 1  # ????????????????????????
        except AttributeError:
            main.count = 1  # ?????????????????????

        print("main%d" % main.count)
        # grab the next frame from the video file
        (grabbed, frame) = _vs.read()

        # check to see if we have reached the end of the video file
        if frame is None:
            break
        frame = frame_add_w_and_h(frame)
        #ok, update_bboxes = _multi_tracker.update(frame)
        ok, update_bboxes = MTC.update(frame)
        if (_adjust_switch & (main.count % 5 == 0)) == True:
            print("adjust+yolo")
            crop_people = crop_people_method(update_bboxes, frame)
            adjust_bboxes = detect_people_and_get_adjust_bboxes2(update_bboxes, frame, crop_people)
            # adjust_bboxes = detect_people_and_get_adjust_bboxes1(update_bboxes, frame)
        else:
            print("tracking")
            adjust_bboxes = update_bboxes.copy()
        print(adjust_bboxes)
        if ok:
            for i, newbox in enumerate(adjust_bboxes):
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
                (startX, startY) = p1
                cv2.putText(frame, "preson", (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        cv2.namedWindow("tracking...", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("tracking...", _frame_size_width, _frame_size_height)
        cv2.imshow("tracking...", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # update the FPS counter
        _fps.update()

    # stop the timer and display FPS information
    _fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(_fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(_fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
    _vs.release()


def show_frame(frame, window_name):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, _frame_size_width, _frame_size_height)
    cv2.imshow(window_name, frame)
    # cv2.destroyWindow(window_name)
    while True:
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break


def show_original_bboxes_and_adjust_bboxes_at_same_frame(frame, user_draw_bboxes, adjust_bboxes):
    draw_frame = frame.copy()

    for i, newbox in enumerate(user_draw_bboxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(draw_frame, p1, p2, (0, 255, 0), 2)
        (startX, startY) = p1
        cv2.putText(draw_frame, "uesr", (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    for i, newbox in enumerate(adjust_bboxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(draw_frame, p1, p2, (0, 0, 255), 2)
        (startX, startY) = p1
        cv2.putText(draw_frame, "model", (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


if __name__ == '__main__':
    # global variables add _ in front of variable
    _adjust_switch = True
    # construct the argument parser and parse the arguments
    _args = read_user_input_info()
    _frame_size_width = 1280
    _frame_size_height = 720
    # initialize the list of class labels MobileNet SSD was trained to
    # detect
    labelsPath = os.path.sep.join([_args["yolo"], "coco.names"])
    _LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(_LABELS), 3),
                               dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([_args["yolo"], "yolov3.weights"])
    configPath = os.path.sep.join([_args["yolo"], "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    _net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    # load our serialized model from disk
    print("[INFO] loading model...")
    _ln = _net.getLayerNames()
    _ln = [_ln[i[0] - 1] for i in _net.getUnconnectedOutLayers()]
    if _args["use_gpu"] == 1:
        # set CUDA as the preferable backend and target
        print("[INFO] setting preferable backend and target to CUDA...")
        _net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        _net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # initialize the video stream and output video writer
    print("[INFO] starting video stream...")
    _vs = cv2.VideoCapture(_args["video"])

    # for saving tracker objects
    _multi_tracker = cv2.MultiTracker_create()

    (grabbed, _frame) = _vs.read()

    print("frame_size:")
    print(_frame.shape[:2])
    _frame = frame_add_w_and_h(_frame)

    # using ROI method to draw bbox
    user_draw_bboxes = detect_people_by_ROI(_frame)

    _detect_method = 3
    if _adjust_switch == True:
        # for person recognition
        start_time = time.time()
        if _detect_method == 1:
            frame_only_one_person = out_of_bbox_range_frame_paint_black(user_draw_bboxes, _frame)
            adjust_bboxes = detect_people_and_get_adjust_bboxes_for_first_frame_paint_black(_frame, user_draw_bboxes,
                                                                                            frame_only_one_person)
            elapsed_time = time.time() - start_time
            print('paint_black_method frist frame adjusted,elapsed time: %2f sec.' % elapsed_time)
        elif _detect_method == 2:
            crop_people = crop_people_method(user_draw_bboxes, _frame)
            adjust_bboxes = detect_people_and_get_adjust_bboxes_for_first_frame_crop_people(_frame, user_draw_bboxes,
                                                                                            crop_people)
            elapsed_time = time.time() - start_time
            print('crop_mathod frist frame adjusted,elapsed time: %2f sec.' % elapsed_time)
        elif _detect_method == 3:
            elapsed_time = time.time() - start_time
            adjust_bboxes = detect_people_and_get_adjust_bboxes_for_first_frame_all_frame(_frame, user_draw_bboxes)
            print('crop_mathod frist frame adjusted,elapsed time: %2f sec.' % elapsed_time)
        print("user_draw_bboxes-------------------------------")
        print(user_draw_bboxes)
        print("adjust_bboxes----------------------------------")
        print(adjust_bboxes)
        show_original_bboxes_and_adjust_bboxes_at_same_frame(_frame, user_draw_bboxes, adjust_bboxes)
        # init_tracker(adjust_bboxes, _frame)
        MTC = mtc.mot_class(_frame,  tuple(map(tuple, user_draw_bboxes)))
        process_task_num = MTC.read_process_task_num()
    else:
        init_tracker( user_draw_bboxes, _frame)

    # start the frames per second throughput estimator
    _fps = FPS().start()

    # tracking person on the video
    main()
