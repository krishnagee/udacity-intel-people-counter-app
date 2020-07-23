import cv2
import argparse
import logging as log
import sys
from inference_v1 import Network
import os
import time
import json


class Input:
    def __init__(self, args):
        self.network = Network()
        self.model = args.model
        self.device = args.device
        self.input = args.input
        self.probability = args.prob_threshold
        self.input_image = None
        self.start_time = 0
        self.prev_count = 0
        self.current_count = 0
        self.frame_count = 0
        self.total_count = 0
        self.duration = 0

    def load_model(self):
        return self.network.load_model(self.model, self.device, 1, 1, 0)[1]

    def clean_network(self):
        self.network.clean()

    def process_input(self):
    # Handle the input stream
        if self.input == 'CAM':  # Check for live feed
            input_stream = 0

        elif self.input.endswith('.jpg') or self.input.endswith('.bmp'):
            self.input_image = True
            input_stream = self.input
            assert os.path.isfile(self.input), "File does not exists"

        else:  # Check for video file
            input_stream = self.input
            assert os.path.isfile(self.input), "File does not exists"
        return input_stream,self.input_image


    def process_asyncrequests(self,image,frame,output,mqtt):
        # Start asynchronous inference for specified request
        inf_start = time.time()
        self.network.exec_net(0, image)

        # Wait for the result
        if self.network.wait(0) == 0:
            det_time = time.time() - inf_start
            # Get the results of the inference request
            result = self.network.output(0)
            # Draw Bounding Box
            frame, self.current_count = output.crop_frame(result, frame, self.frame_count, self.prev_count)

            # Printing Inference Time
            fps = 1. / det_time
            delta_count = self.current_count - self.prev_count
            log.info(delta_count)
            # publish data to mqtt mosca server
            # Calculate and send relevant information
            if delta_count > 0:
                start_time = time.time()
                self.total_count = self.total_count + delta_count
                # publish data to MQTT MOSCA server
                mqtt.publishtomqtt("person", json.dumps({"total": self.total_count}))

            elif delta_count < 0:
                self.duration = int(time.time() - self.start_time)
                # publish data to MQTT MOSCA server
                mqtt.publishtomqtt("person/duration", json.dumps({"duration": self.duration}))

            mqtt.publishtomqtt("person", json.dumps({"count": self.current_count}))
            self.prev_count = self.current_count
        return self.current_count, self.prev_count, self.duration, det_time,fps

    def preprocess_image(self,frame, n, c, h, w):
        image = cv2.resize(frame, (w, h))
    # Change data layout from HWC to CHW
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        return image
