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


import cv2
import argparse
from mqtt import Mqtt
from processor import Input
from post_process import Output



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


def infer_on_stream(args, mqtt):
    # Initialise the Processor class
    input = Input(args)
    # Step1 : Load the model
    n, c, h, w = input.load_model()
    # Step2 : Process input stream
    input_stream, input_image = input.process_input()

    # Step3 : Capture Input video stream
    cap = cv2.VideoCapture(input_stream)
    initial_width = cap.get(3)
    initial_height = cap.get(4)

    # Initialize post processor
    output = Output(initial_width, initial_height)

    # Loop until stream is over
    while cap.isOpened():
        # Read from the video capture
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        # Image pre processing using OpenCV apis
        # Start async inference
        image = input.preprocess_image(frame, n, c, h, w)

        current_count, prev_count, duration, det_time, fps = input.process_asyncrequests(image, frame, output, mqtt)

        current_stats = "Current count: %d " % current_count
        output.write_cvtext(frame, current_stats, 15, 45, (255, 0, 255))

        inf_time_message = "FPS: {} , Inference time: {:.3f}ms".format(round(fps, 2), round(det_time * 1000))
        output.write_cvtext(frame, inf_time_message, 15, 15, (255, 0, 255))

        output.send_alert(frame, current_count - prev_count, duration)

        if key_pressed == 27:
            break

        output.write_ffmpeg(frame)

        # Save the Image
        if input_image:
            cv2.imwrite('output_image.jpg', frame)

    cap.release()
    cv2.destroyAllWindows()
    input.clean_network()


def main():
    # Fetch the arguments from command line
    args = build_argparser().parse_args()
    # Connect to the MQTT server and return MQTT client
    mqtt = Mqtt()
    mqtt.connect()
    # Perform inference on the input stream
    infer_on_stream(args, mqtt)
    mqtt.disconnect()


if __name__ == '__main__':
    main()
    exit(0)
