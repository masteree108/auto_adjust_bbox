from imutils.video import FPS
from multiprocessing import Pool
import numpy as np
import argparse
import imutils
import cv2
import os
import time
import mot_class as mtc

def read_user_input_info():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=True,
                    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
                    help="path to Caffe pre-trained model")
    ap.add_argument("-v", "--video", required=True,
                    help="path to input video file")
    ap.add_argument("-o", "--output", type=str,
                    help="path to optional output video file")
    ap.add_argument("-c", "--confidence", type=float, default=0.2,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-u", "--use-gpu", type=int, default=0,
                    help="boolean indicating if CUDA GPU should be used")
    args = vars(ap.parse_args())

    return args


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

        offest_x = int(w/5)
        offest_y = int(h/5)

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
        img_CMB = cv2.copyMakeBorder(crop_img, yt, vh-yb, xl, vw-xr, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        frame_only_one_person.append(img_CMB)                                                                                                               
        #cv2.imwrite("user_pick_" + str(i) + "_" + str(time.time())+"_.png", img_CMB)

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

        offest_x = int(w/5)
        offest_y = int(h/5)

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
        #cv2.imwrite("crop_" + str(i) + "_" + str(time.time())+"_.png", crop_img)

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
            return adjust_bboxes[index]
        else:
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

        interArea = max(0, xB-xA + 1) * max(0, yB-yA + 1)
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
        blob = cv2.dnn.blobFromImage(person, 0.007843, (w, h), 127.5)
        _net.setInput(blob)
        detections = _net.forward()
        recog_person = False
        for j in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, j, 2]

            if confidence > _args["confidence"]:
                idx = int(detections[0, 0, j, 1])
                label = _CLASSES[idx]
                # print("label:%s" % label)
                if _CLASSES[idx] != "person":
                    continue
                else:
                    recog_person = True
                    print("%d recog_person" % i)
                    recog_bbox = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = recog_bbox.astype("int")
                    print("startX:%d" % startX)
                    print("startY:%d" % startY)
                    print("endX:%d" % endX)
                    print("endY:%d" % endY)

                    wadj = int(abs(endX - startX))
                    offset_w = int(wadj/3)
                    wadj = wadj+offset_w
                    print("width_adjust:%d" % wadj)
                    hadj = int(abs(endY - startY))
                    offset_h = int(hadj/10)
                    hadj = hadj+offset_h
                    print("height_adjust:%d" % hadj)
                    get_bboxes[i].append((startX, startY, wadj, hadj))

        if recog_person == False:
            final_bboxes.append(user_draw_bboxes[i])
        else:
            # IoU check which box is best
            final_bboxes.append(IOU_check_for_first_frame(user_draw_bboxes[i], get_bboxes[i]))

    return final_bboxes

def detect_people_and_get_adjust_bboxes_for_first_frame_all_frame(frame, user_draw_bboxes):
    final_bboxes = []
    get_bboxes = []


    # 1. detect people on this frame
    detect_bboxes = []
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)
    _net.setInput(blob)
    detections = _net.forward()
    recog_person = False
    for j in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, j, 2]

        if confidence > _args["confidence"]:
            idx = int(detections[0, 0, j, 1])
            label = _CLASSES[idx]
            # print("label:%s" % label)
            if _CLASSES[idx] != "person":
                continue
            else:
                recog_person = True
                recog_bbox = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = recog_bbox.astype("int")
                print("startX:%d" % startX)
                print("startY:%d" % startY)
                print("endX:%d" % endX)
                print("endY:%d" % endY)

                wadj = int(abs(endX - startX))
                offset_w = int(wadj/3)
                wadj = wadj+offset_w
                print("width_adjust:%d" % wadj)
                hadj = int(abs(endY - startY))
                offset_h = int(hadj/10)
                hadj = hadj+offset_h
                print("height_adjust:%d" % hadj)
                detect_bboxes.append((startX, startY, wadj, hadj))

    # debug:watch detect result 
    save_result = False
    sframe = frame.copy()
    if save_result == True:
        for i, newbox in enumerate(detect_bboxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(sframe, p1, p2, (0, 0, 255), 2)
            (startX, startY) = p1
            cv2.putText(sframe, "model", (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.imwrite("detect_result" + "_" + str(time.time())+"_.png", sframe)

    # 2. using IoU to find which are match user drawed bboxes
    for i,bbox in enumerate(user_draw_bboxes):
        # IoU check which box is best
        final_bboxes.append(IOU_check_for_first_frame(bbox, detect_bboxes))

    return final_bboxes

def detect_people_and_get_adjust_bboxes_for_first_frame_crop_people(frame, user_draw_bboxes, crop_people):
    final_bboxes = []
    get_bboxes = []

    for i, person in enumerate(crop_people):
        get_bboxes.append([])
        (h, w) = person.shape[:2]
        blob = cv2.dnn.blobFromImage(person, 0.007843, (w, h), 127.5)
        _net.setInput(blob)
        detections = _net.forward()
        recog_person = False
        for j in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, j, 2]

            if confidence > _args["confidence"]:
                idx = int(detections[0, 0, j, 1])
                label = _CLASSES[idx]
                # print("label:%s" % label)
                if _CLASSES[idx] != "person":
                    continue
                else:
                    recog_person = True
                    print("%d recog_person" % i)
                    recog_bbox = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = recog_bbox.astype("int")
                    print("startX:%d" % startX)
                    print("startY:%d" % startY)
                    print("endX:%d" % endX)
                    print("endY:%d" % endY)

                    wadj = int(abs(endX - startX))
                    offset_w = int(wadj/3)
                    wadj = wadj+offset_w
                    print("width_adjust:%d" % wadj)
                    hadj = int(abs(endY - startY))
                    offset_h = int(hadj/8)
                    hadj = hadj+offset_h
                    print("height_adjust:%d" % hadj)
                    adjust_x = user_draw_bboxes[i][0] + startX
                    adjust_y = user_draw_bboxes[i][1] + startY
                    get_bboxes[i].append((adjust_x, adjust_y, wadj, hadj))

        if recog_person == False:
            final_bboxes.append(user_draw_bboxes[i])
        else:
            # IoU check which box is best
            final_bboxes.append(IOU_check_for_first_frame(user_draw_bboxes[i], get_bboxes[i]))

    return final_bboxes


def detect_people_and_get_adjust_bboxes2(bboxes, frame, crop_people):
    final_bboxes = []
    get_bboxes = []

    for i, crop_person in enumerate(crop_people):
        ct = i - 1
        '''
        # test only adjust below
        if i % 2 == 0:
            final_bboxes.append(bboxes[ct])
            return final_bboxes 
        '''
        (h, w) = frame.shape[:2]
        # (h, w) = crop_person.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)
        # blob = cv2.dnn.blobFromImage(crop_person, 0.007843, (w, h), 127.5)
        _net.setInput(blob)
        detections = _net.forward()
        recog_person = False
        for j in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, j, 2]

            if confidence > _args["confidence"]:
                idx = int(detections[0, 0, j, 1])
                label = _CLASSES[idx]
                # print("label:%s" % label)
                if _CLASSES[idx] != "person":
                    continue
                else:
                    recog_person = True
                    print("recog_person")
                    recog_bbox = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = recog_bbox.astype("int")
                    print("startX:%d" % startX)
                    print("startY:%d" % startY)
                    print("endX:%d" % endX)
                    print("endY:%d" % endY)
                    wadj = int(abs(endX - startX))
                    # wadj = wadj + int(wadj/10)
                    print("width_adjust:%d" % wadj)
                    hadj = int(abs(endY - startY))
                    # hadj = hadj + int(hadj/10)
                    print("height_adjust:%d" % hadj)
                    x = bboxes[ct][0] + startX
                    y = bboxes[ct][1] + startY
                    # final_bboxes.append((x, y, wadj, hadj))
                    get_bboxes.append((startX, startY, wadj, hadj))
        if recog_person == False:
            # final_bboxes.append((0 ,0 ,0 ,0))
            final_bboxes.append(bboxes[ct])
        else:
            # IoU check which box is best
            final_bboxes.append(IOU_check(bboxes[ct], get_bboxes))

    return final_bboxes


def frame_add_w_and_h(frame):
    (vh, vw) = frame.shape[:2]
    cmb_w = 0
    cmb_h = 0
    do_border_w = True
    do_border_h = True
    if vw < _frame_size_width :
        cmb_w = int(abs(_frame_size_width - vw) / 2)
        do_border_w = False

    if vh < _frame_size_height:
        cmb_h = int(abs(_frame_size_height  - vh) / 2)
        do_border_h = False

    if do_border_w == True and do_border_h == True:
        img_CMB = cv2.copyMakeBorder(frame, cmb_h, cmb_h, cmb_w, cmb_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return img_CMB
    else:
        return frame

def main():
    # loop over frames from the video file stream
    while True:
        # grab the next frame from the video file
        (grabbed, frame) = _vs.read()

        # check to see if we have reached the end of the video file
        if frame is None:
            break

        frame = frame_add_w_and_h(frame)
        ok, update_bboxes = MTC.update(frame)
        if _adjust_switch == True:
            print("adjust")
            crop_people = crop_those_people(update_bboxes, frame)
            adjust_bboxes = detect_people_and_get_adjust_bboxes2(update_bboxes, frame, crop_people)
            # adjust_bboxes = detect_people_and_get_adjust_bboxes1(update_bboxes, frame)
        else:
            adjust_bboxes = update_bboxes.copy()
        print(adjust_bboxes)
        if ok:
            for i, newbox in enumerate(adjust_bboxes):
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[2]), int(newbox[3]))
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
    #cv2.destroyWindow(window_name)
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

    show_frame(draw_frame, "user draws bboxes(green) and model adjust bboxes(red)")

if __name__ == '__main__':
    # global variables add _ in front of variable
    _adjust_switch = False
    # construct the argument parser and parse the arguments
    _args = read_user_input_info()
    _frame_size_width = 1280
    _frame_size_height = 720
    # initialize the list of class labels MobileNet SSD was trained to
    # detect
    _CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                "sofa", "train", "tvmonitor"]



    # load our serialized model from disk
    print("[INFO] loading model...")
    _net = cv2.dnn.readNetFromCaffe(_args["prototxt"], _args["model"])
    if _args["use_gpu"] == 1:
        # set CUDA as the preferable backend and target
        print("[INFO] setting preferable backend and target to CUDA...")
        _net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        _net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


    # initialize the video stream and output video writer
    print("[INFO] starting video stream...")
    _vs = cv2.VideoCapture(_args["video"])

    (grabbed, _frame) = _vs.read()

    print("frame_size:")
    print(_frame.shape[:2])
    _frame = frame_add_w_and_h(_frame)

    # using ROI method to draw bbox
    user_draw_bboxes = detect_people_by_ROI(_frame)

    _detect_method = 1
    if _adjust_switch == True:
        # for person recognition
        start_time = time.time()
        if _detect_method == 1:
            frame_only_one_person = out_of_bbox_range_frame_paint_black(user_draw_bboxes, _frame)
            adjust_bboxes = detect_people_and_get_adjust_bboxes_for_first_frame_paint_black(_frame, user_draw_bboxes, frame_only_one_person)
            elapsed_time = time.time() - start_time
            print('paint_black_method frist frame adjusted,elapsed time: %2f sec.' % elapsed_time)
        elif _detect_method == 2:
            crop_people = crop_people_method(user_draw_bboxes, _frame)
            adjust_bboxes = detect_people_and_get_adjust_bboxes_for_first_frame_crop_people(_frame, user_draw_bboxes, crop_people)
            elapsed_time = time.time() - start_time
            print('crop_mathod frist frame adjusted,elapsed time: %2f sec.' % elapsed_time)
        elif _detect_method == 3:
            elapsed_time = time.time() - start_time
            adjust_bboxes = detect_people_and_get_adjust_bboxes_for_first_frame_all_frame(_frame, user_draw_bboxes)
            print('crop_mathod frist frame adjusted,elapsed time: %2f sec.' % elapsed_time)

        print(user_draw_bboxes)
        print(adjust_bboxes)
        show_original_bboxes_and_adjust_bboxes_at_same_frame(_frame, user_draw_bboxes, adjust_bboxes)
        MTC = mtc.mot_class(_frame,  adjust_bboxes)
    else:
        MTC = mtc.mot_class(_frame,  user_draw_bboxes)

    # start the frames per second throughput estimator
    _fps = FPS().start()

    # tracking person on the video
    main()
