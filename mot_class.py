# plseas note this program uses multi-processing module, so it can not using multi-threading here!!
# watch CPU loading on the ubuntu
# ps axu | grep [M]OTF_process_pool.py | awk '{print $2}' | xargs -n1 -I{} ps -o sid= -p {} | xargs -n1 -I{} ps --forest -o user,pid,ppid,cpuid,%cpu,%mem,stat,start,time,command -g {}

# import the necessary packages
import multiprocessing
import numpy as np
import cv2
import os


class mot_class():
    # private

    # for saving tracker objects
    __detect_amount_of_people = 0
    __processor_task_num = []

    def __get_algorithm_tracker(self, algorithm):
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

    def __assign_amount_of_people_for_tracker(self, detect_amount_of_people, using_processor_qty):
        # it should brings (left, top, width, height) to tracker.init() function
        # parameters are left, top , right and bottom in the box
        # so those parameters need to minus like below to get width and height
        left_num = detect_amount_of_people % using_processor_qty
        process_num = int(detect_amount_of_people / using_processor_qty)
        processor_task_num = []
        process_num_ct = 0
        # print("bboxes:")
        # print(bboxes)
        for i in range(using_processor_qty):
            task_ct = 0
            tracker = cv2.MultiTracker_create()
            for j in range(process_num_ct, process_num_ct + process_num):
                task_ct = task_ct + 1
                process_num_ct = process_num_ct + 1
            processor_task_num.append(task_ct)
        if left_num != 0:
            counter = 0
            k = detect_amount_of_people - using_processor_qty * process_num
            for k in range(k, k + left_num):
                # print("k:%d" % k)
                processor_task_num[counter] = processor_task_num[counter] + 1
                counter = counter + 1
                # print("processor_task_number:")
        # print(processor_task_num)
        return processor_task_num

    # public
    def __init__(self, frame, bboxes):

        self.inputQueues = []
        self.outputQueues = []

        self.__detect_amount_of_people = len(bboxes)
        self.__using_processor_qty = 0
        core_for_os = 1
        if self.__detect_amount_of_people >= (os.cpu_count() - core_for_os):
            self.__using_processor_qty = os.cpu_count() - core_for_os
        else:
            self.__using_processor_qty = self.__detect_amount_of_people

        self.__processor_task_num = self.__assign_amount_of_people_for_tracker(self.__detect_amount_of_people,
                                                                               self.__using_processor_qty)
        ct = 0;
        for i in range(self.__using_processor_qty):
            bboxes_for_trackers = []
            for j in range(int(self.__processor_task_num[i])):
                bboxes_for_trackers.append(bboxes[ct])
                ct += 1
            # print("===================================")
            # print(bboxes_for_trackers)
            iq = multiprocessing.Queue()
            oq = multiprocessing.Queue()
            self.inputQueues.append(iq)
            self.outputQueues.append(oq)

            processes = multiprocessing.Process(
                target=self.tracker_process,
                args=(frame, bboxes_for_trackers, iq, oq))
            processes.daemon = True
            processes.start()

            # print("detect_amount_of_people: %d" % self.__detect_amount_of_people)
            # print("processor_task_num")
            # print(self.__processor_task_num)

    def tracker_process(self, frame, bboxes, inputQueue, outputQueue):
        # print("tracker_process")
        tracker = cv2.MultiTracker_create()
        for i, bbox in enumerate(bboxes):
            print("bbox--------------------------------------")
            print(bbox)
            tracker.add(cv2.TrackerCSRT_create(), frame, bbox)

        while True:
            bboxes_org = []
            bboxes_transfer = []
            frame = inputQueue.get()
            # print("receive frame")
            ok, bboxes_org = tracker.update(frame)
            # print(bboxes_org)
            for box in bboxes_org:
                startX = box[0]
                startY = box[1]
                endX = box[0] + box[2]
                endY = box[1] + box[3]
                bbox = (startX, startY, box[2], box[3])
                bboxes_transfer.append(bbox)
            outputQueue.put(bboxes_transfer)

    def read_process_task_num(self):
        return self.__processor_task_num

    # tracking person on the video
    def update(self, frame):
        for i, iq in enumerate(self.inputQueues):
            iq.put(frame)

        bboxes_temp = []
        for i, oq in enumerate(self.outputQueues):
            bboxes_temp.append(oq.get())

        # print(bboxes_temp)
        bboxes = []
        # for i in range(self.__using_processor_qty):
        for i, bbox in enumerate(bboxes_temp):
            for j in range(self.__processor_task_num[i]):
                # print(j)
                bboxes.append(bbox[j])

        # print(bboxes)
        # print("len(bboxes):%d" % len(bboxes))

        if len(bboxes) > 0:
            return True, bboxes
        else:
            return False, bboxes

