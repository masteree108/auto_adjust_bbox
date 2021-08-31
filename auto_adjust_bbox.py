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
    args = vars(ap.parse_args())

    return args

def init_tracker(detect_people_num, box, frame):
    #print("detect_people_,number:%d" % detect_people_num)
    # grab the appropiate object tracker using our dictionary of 
    # OpenCV object tracker objects
    # it should brings (left, top, width, height) to tracker.init() function
    # parameters are left, top , right and bottom in the box 
    # so those parameters need to minus like below to get width and height 
    bbox = (box[0], box[1], abs(box[0]-box[2]), abs(box[1]-box[3]))
    _multi_tracker.add(get_algorithm_tracker("CSRT"), frame, bbox) 

def crop_those_people(bboxes, frame):
    offset = 5
    (vh, vw) = _frame.shape[:2]
    for i,bbox in enumerate(bboxes):
        x = int(bbox[0])
        y = int(bbox[1])
        w = int(bbox[2])
        h = int(bbox[3])

        xl = x - offset
        if xl < 0:
            xl = 0

        xr = x + w + offset
        if xr > vw:
            xr = vw

        yt = y - offset
        if yt < 0:
            yt = 0

        yb = y + h + offset
        if yb > vh:
            yb = vh

        crop_img = frame[yt:yb, xl:xr]
        _crop_people.append(crop_img)
        cv2.imwrite(str(i)+".png", crop_img)

def detect_people_by_ROI_and_tracker_init(frame, w, h):
    ROI_window_name = "multi ROIs(draw bbox ok:Enter or Space, exit:Esc)"
    cv2.namedWindow(ROI_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ROI_window_name, w, h)
    # bbox:posX ,posY, width, length
    bboxes = cv2.selectROIs(ROI_window_name, frame, False)
    cv2.destroyWindow(ROI_window_name)
    for i,bbox in enumerate(bboxes):
        newbbox = []
        newbbox.append(int(bbox[0]))
        newbbox.append(int(bbox[1]))
        newbbox.append(int(bbox[0]) + int(bbox[2]))
        newbbox.append(int(bbox[1]) + int(bbox[3]))
        init_tracker(i, newbbox, frame)

    return bboxes 

def detect_people_and_get_bbox():
    # detecting how many person on this frame
    detect_people_qty = 0
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:
            idx = int(detections[0, 0, i, 1])
            label = _CLASSES[idx]
            #print("label:%s" % label)
            if _CLASSES[idx] != "person":
                continue
                
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            bb = (startX, startY, endX, endY)
            #print(bb)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            #print("label:%s" % label)
            cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            init_tracker(detect_people_qty, bb, frame)
            detect_people_qty = detect_people_qty + 1 
    print("detect_people_quantity:%d" % detect_people_qty)
    return detect_people_qty

def main():
    # loop over frames from the video file stream
    while True:
	# grab the next frame from the video file
        (grabbed, frame) = _vs.read()

	    # check to see if we have reached the end of the video file
        if frame is None:
            break
        
        #frame = imutils.resize(frame, width=frame_size_width)
        ok, bboxes = _multi_tracker.update(frame)
        if ok:                                         
            for i, newbox in enumerate(bboxes):
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
                (startX, startY) = p1
                cv2.putText(frame, "preson", (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
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

if __name__ == '__main__':
    # global variables add _ in front of variable

    # construct the argument parser and parse the arguments
    _args = read_user_input_info()
    _crop_people = []

    # initialize the list of class labels MobileNet SSD was trained to
    # detect
    _CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	    "sofa", "train", "tvmonitor"]

    # load our serialized model from disk
    print("[INFO] loading model...")
    _net = cv2.dnn.readNetFromCaffe(_args["prototxt"], _args["model"])

    # initialize the video stream and output video writer
    print("[INFO] starting video stream...")
    _vs = cv2.VideoCapture(_args["video"])
    
    # for saving tracker objects
    _multi_tracker = cv2.MultiTracker_create()

    (grabbed, _frame) = _vs.read()
    #frame = imutils.resize(frame, width=frame_size_width)
    print("frame_size:")
    print(_frame.shape[:2])
    (h, w) = _frame.shape[:2]
    blob = cv2.dnn.blobFromImage(_frame, 0.007843, (w, h), 127.5)
    _net.setInput(blob)
    _detections = _net.forward()
    
    # using ROI method to draw bbox
    bboxes = detect_people_by_ROI_and_tracker_init(_frame, w, h)

    # for person recognition recognition
    crop_those_people(bboxes, _frame)

    # auto adjust bbox size 

    # start the frames per second throughput estimator
    _fps = FPS().start()

    # tracking person on the video
    main()

