import cv2
import argparse
import logging as log
import sys


class Output:
    def __init__(self, width, height):
        self.count = None
        self.total = None
        self.duration = None
        self.initial_width = width
        self.initial_height = height

    def crop_frame(self, coord, frame, frame_count, prev_count):
        current_count = 0

        for obj in coord[0][0]:
            # Normalized bounding box
            if obj[2] > 0.80:
                xmin = int(obj[3] * self.initial_width)
                ymin = int(obj[4] * self.initial_height)
                xmax = int(obj[5] * self.initial_width)
                ymax = int(obj[6] * self.initial_height)

                if obj[1] == 1:
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 255), 1)
                    current_count = current_count + 1
                    # distance_center = math.sqrt(math.pow(x_center - center_x, 2) + math.pow(y_center - center_y, 2) * 1.0)
                    # frame_count = 0
                    log.info("current frame count", current_count)
        return frame, current_count

    def send_alert(self, frame, delta_count, duration):
        warning_message = ""
        if delta_count >= 5:
            warning_message = "Maximum supported images per frame limit reached"
        elif duration > 300000:
            warning_message = "Person is staying in store for long time"
        (text_width, text_height) = cv2.getTextSize(warning_message, cv2.QT_FONT_NORMAL, 0.5, thickness=1)[0]
        text_offset_x = 10
        text_offset_y = frame.shape[0] - 10
        # make the co-ordinates of the box with a small padding of two pixels
        box_coord = ((text_offset_x, text_offset_y + 2), (text_offset_x + text_width, text_offset_y - text_height - 2))
        cv2.rectangle(frame, box_coord[0], box_coord[1], (0, 0, 0), cv2.FILLED)
        self.write_cvtext(frame, warning_message, text_offset_x, text_offset_y, (0, 0, 255))

    def write_ffmpeg(self, frame):
        # Send the frame to the FFMPEG server
        frame = cv2.resize(frame, (768, 432))
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

    def write_cvtext(self, frame, text, X, Y, color):
        cv2.putText(frame, text, (X, Y), cv2.QT_FONT_NORMAL, 0.5, color, 1)
