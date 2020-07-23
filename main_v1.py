"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import time
import json
import cv2
import argparse
import logging as log
from inference import Network
from mqtt import Mqtt
import math


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = argparse.ArgumentParser(description='Basic OpenVINO Example with MobileNet-SSD')
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.55,
                        help="Probability threshold for detections filtering"
                             "(0.55 by default)")
    return parser


def crop_frame(coord, frame, frame_count, prev_count):
    current_count = 0
    # center_x = frame.shape[1] / 2
    # center_y = frame.shape[0] / 2
    # distance_center = 0
    for obj in coord[0][0]:
        # Normalized bounding box
        if obj[2] > 0.80:
            xmin = int(obj[3] * initial_width)
            ymin = int(obj[4] * initial_height)
            xmax = int(obj[5] * initial_width)
            ymax = int(obj[6] * initial_height)
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            if obj[1] == 1:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 255), 1)
                current_count = current_count + 1
                # distance_center = math.sqrt(math.pow(x_center - center_x, 2) + math.pow(y_center - center_y, 2) * 1.0)
                # frame_count = 0

                log.info("current frame count", current_count)
        # if current_count == 0:
        #     frame_count +=1
        #
        # if distance_center > 0 and frame_count < 5:
        #     current_count = 1
        #     frame_count += 1
        #     if frame_count > 100:
        #         frame_count = 0
    return frame, current_count


def send_alert(frame, delta_count, duration):
    warning_message = ""
    if delta_count >= 5:
        warning_message = "Maximum supported images per frame limit reached"
    elif duration > 3000:
        warning_message = "Person is staying in store for long time"
    (text_width, text_height) = cv2.getTextSize(warning_message, cv2.QT_FONT_NORMAL, 0.5, thickness=1)[0]
    text_offset_x = 10
    text_offset_y = frame.shape[0] - 10
    # make the co-ordinates of the box with a small padding of two pixels
    box_coord = ((text_offset_x, text_offset_y + 2), (text_offset_x + text_width, text_offset_y - text_height - 2))
    cv2.rectangle(frame, box_coord[0], box_coord[1], (0, 0, 0), cv2.FILLED)
    write_cvtext(frame, warning_message, text_offset_x, text_offset_y, (0, 0, 255))


def infer_person(args, mqtt):
    # Initialise the IE network plugin class
    infer_network = Network()
    # Initialize the IE model name
    model = args.model
    # Initialize the video feed or camera feed
    video_file = args.input
    # Initialize the device either CPU, GPU , FPGA , VDU or HDDL
    device = args.device

    # Flag for the input image
    image_flag = False

    start_time = 0
    prev_count = 0
    total_count = 0
    duration = 0

    # Load the model through IE plugin
    n, c, h, w = infer_network.load_model(model, device, 1, 1, 0)[1]

    # Handle the input stream
    if video_file == 'CAM':  # Check for live feed
        input_stream = 0
    # Check for input image
    elif video_file.endswith('.jpg') or video_file.endswith('.bmp'):
        image_flag = True
        input_stream = video_file
        assert os.path.isfile(video_file), "Specified input file doesn't exist"

    else:  # Check for video file
        input_stream = video_file
        assert os.path.isfile(video_file), "Specified input file doesn't exist"

    cap = cv2.VideoCapture(input_stream)
    global initial_width, initial_height, prob_threshold, frame_count
    frame_count = 0
    initial_width = cap.get(3)
    initial_height = cap.get(4)

    # Initialize the probability value
    prob_threshold = args.prob_threshold

    # Loop until stream is over
    while cap.isOpened():
        # Read from the video capture
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        # Image pre processing using OpenCV apis
        # Start async inference
        image = preprocess_image(frame, n, c, h, w)
        # Start asynchronous inference for specified request
        inf_start = time.time()
        infer_network.exec_net(0, image)

        # Wait for the result
        if infer_network.wait(0) == 0:
            det_time = time.time() - inf_start
            # Get the results of the inference request
            result = infer_network.output(0)
            # Draw Bounding Box
            frame, current_count = crop_frame(result, frame, frame_count, prev_count)

            # Printing Inference Time
            fps = 1. / det_time
            delta_count = current_count - prev_count
            log.info(delta_count)
            # publish data to mqtt mosca server
            # Calculate and send relevant information
            if delta_count > 0:
                start_time = time.time()
                total_count = total_count + delta_count
                # publish data to MQTT MOSCA server
                mqtt.publishtomqtt("person", json.dumps({"total": total_count}))

            elif delta_count < 0:
                duration = int(time.time() - start_time)
                # publish data to MQTT MOSCA server
                mqtt.publishtomqtt("person/duration", json.dumps({"duration": duration}))

            current_stats = "Current count: %d " % current_count
            write_cvtext(frame, current_stats, 15, 45, (255, 0, 255))

            inf_time_message = "FPS: {} , Inference time: {:.3f}ms".format(round(fps, 2), round(det_time * 1000))
            write_cvtext(frame, inf_time_message, 15, 15, (255, 0, 255))

            mqtt.publishtomqtt("person", json.dumps({"count": current_count}))
            send_alert(frame, delta_count, duration)
            prev_count = current_count

            if key_pressed == 27:
                break

        write_ffmpeg(frame)

        # Save the Image
        if image_flag:
            cv2.imwrite('output_image.jpg', frame)

    cap.release()
    cv2.destroyAllWindows()
    infer_network.clean()


def write_ffmpeg(frame):
    # Send the frame to the FFMPEG server
    frame = cv2.resize(frame, (768, 432))
    sys.stdout.buffer.write(frame)
    sys.stdout.flush()


def write_cvtext(frame, text, X, Y, color):
    cv2.putText(frame, text, (X, Y), cv2.QT_FONT_NORMAL, 0.5, color, 1)


def preprocess_image(frame, n, c, h, w):
    image = cv2.resize(frame, (w, h))
    # Change data layout from HWC to CHW
    image = image.transpose((2, 0, 1))
    image = image.reshape((n, c, h, w))
    return image


def main():
    # Fetch the arguments from command line
    args = build_argparser().parse_args()
    # Connect to the MQTT server and return MQTT client
    mqtt = Mqtt()
    mqtt.connect()
    # Perform inference on the input stream
    infer_person(args, mqtt)
    mqtt.disconnect()


if __name__ == '__main__':
    main()
    exit(0)
