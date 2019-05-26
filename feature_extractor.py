"""
Creates a record file composed of feature vectors by detecting objects on screen.
Optionally allows capturing of images for the use in labeling to train the object detection model.

Notes:
    Keyboard module requires elevated privilege in order to run, this is used in making training data.
    The script captures the top left region of the screen of 800x600 (width by height)
    The images are purposefully handled in gray-scale for reduced mem usage and to ignore colors during inference,
        the model was created and trained with gray-scale, so color images have undetermined accuracy.

Written by Triston Scallan
"""
from tensorflow.models.research.object_detection.utils import label_map_util
from tensorflow.models.research.object_detection.utils import visualization_utils as vis_util

import numpy as np
import os
import cv2
import time
import mss
import keyboard
import tensorflow as tf


SHOW_DISPLAY = False                                                    # config for showing the inference to a window.
TRAINING_DIR = 'training_data'
MODEL_DIR = os.path.join('models', 'ssd_mobilenet_v1_audiosurf')
LABEL_DIR = 'data'
IMAGE_DIR = 'images'
PATH_TO_TRAINING_FILE = os.path.join(TRAINING_DIR, 'LSTM_train_data_session1.npy')


# consider converting list types to numpy arrays
class TOD(object):
    def __init__(self):
        self.PATH_TO_CKPT = os.path.join(MODEL_DIR, 'frozen_inference_graph.pb')
        self.PATH_TO_LABELS = os.path.join(LABEL_DIR, 'object-detection.pbtxt')
        self.detection_graph = self._load_model()
        self.box_tensor, self.score_tensor, self.class_tensor, self.image_tensor = self._load_tensors()
        self.category_index = self._load_label_map()
        self.last_box_center = [0, 0, 0, 0, 0, 1, 0]

    def _load_model(self):
        with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fid.read())
        with tf.Graph().as_default() as detection_graph:
            tf.import_graph_def(graph_def, name='')
        return detection_graph

    def _load_label_map(self):
        return label_map_util.create_category_index_from_labelmap(self.PATH_TO_LABELS, use_display_name=True)

    def _load_tensors(self):
        return [self.detection_graph.get_tensor_by_name('detection_boxes:0'),
                self.detection_graph.get_tensor_by_name('detection_scores:0'),
                self.detection_graph.get_tensor_by_name('detection_classes:0'),
                self.detection_graph.get_tensor_by_name('image_tensor:0')]

    def detect(self, session: tf.Session):
        (boxes, scores, classes) = session.run(
            (self.box_tensor, self.score_tensor, self.class_tensor),
            feed_dict={self.image_tensor: image_np})

        # filter by score threshold and mask out the lane detections
        lane_score_mask = np.logical_and(np.squeeze(classes) != 4, np.squeeze(scores) > .7)
        boxes = np.squeeze(boxes)[lane_score_mask]
        classes = np.squeeze(classes).astype(np.int32)[lane_score_mask]
        scores = np.squeeze(scores)[lane_score_mask]

        # filter to the box that appears lowest on screen, if none then default to zeros
        if (classes == 1).any():
            blocks = boxes[classes == 1]
            blocks = blocks[blocks[:, 0].argmax()]
        else:
            blocks = np.zeros(4)

        if (classes == 2).any():
            ship = boxes[classes == 2]
            ship = ship[ship[:, 0].argmax()]
        else:
            ship = np.zeros(4)

        if (classes == 3).any():
            spikes = boxes[classes == 3]
            spikes = spikes[spikes[:, 0].argmax()]
        else:
            spikes = np.zeros(4)

        if len(boxes) <= 0:
            feature_vector = self.last_box_center
        else:
            # convert box detection (y_min, x_min, y_max, x_max) -> (x_center, y_center)
            # reduce granularity to the floor of 0.05
            block_vector = [(np.divide(blocks[3] + blocks[1], 2) * 20 // 1 / 20) + 0,
                            (np.divide(blocks[2] + blocks[0], 2) * 20 // 1 / 20) + 0]
            spike_vector = [(np.divide(spikes[3] + spikes[1], 2) * 20 // 1 / 20) + 0,
                            (np.divide(spikes[2] + spikes[0], 2) * 20 // 1 / 20) + 0]
            # convert ship into a one-hot based on it's lane
            if np.any(ship != np.zeros(4)):
                ship_center = np.divide(ship[3] + ship[1], 2)
                if ship_center < 0.375:
                    ship_vector = [1, 0, 0]
                elif ship_center < 0.625:
                    ship_vector = [0, 1, 0]
                else:
                    ship_vector = [0, 0, 1]
            else:
                ship_vector = self.last_box_center[-3:]

            # combine features into a single vector and save it
            feature_vector = block_vector + spike_vector + ship_vector
            self.last_box_center = feature_vector

        # if SHOW_DISPLAY mode is on, SHOW_DISPLAY the detection to the screen. NOTE, this inflates iteration time.
        if SHOW_DISPLAY is True:
            vis_util.visualize_boxes_and_labels_on_image_array(
                screen,
                boxes,
                classes,
                scores,
                self.category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            cv2.namedWindow("detection", cv2.WINDOW_NORMAL)
            cv2.imshow("detection", screen)
        return feature_vector


def check_keys():
    if keyboard.is_pressed('left'):
        keys = [1, 0, 0]
    elif keyboard.is_pressed('right'):
        keys = [0, 0, 1]
    else:
        keys = [0, 1, 0]
    return keys


if __name__ == '__main__':
    if os.path.isfile(PATH_TO_TRAINING_FILE):                           # check file
        print('Training file exists, loading previous data!')
        training_data = list(np.load(PATH_TO_TRAINING_FILE))
    else:
        print('Training file does not exist, starting fresh!')
        training_data = []

    print("press the 1 key to start capture-only mode, press the 2 key to start inference mode. 'q' to quit now.")
    while True:
        try:
            if keyboard.is_pressed('1'):  # wait for the user to hit the start key
                mode = 1
                break
            elif keyboard.is_pressed('2'):
                mode = 2
                break
            elif keyboard.is_pressed('q'):
                quit()
        except RuntimeError as ignored:
            pass

    for i in list(range(4))[::-1]:
        print(i)
        time.sleep(1)

    with mss.mss() as sct:                                              # create resource to capture screen
        region = {'top': 0, 'left': 0, 'width': 800, 'height': 600}     # define region of interest
        detector = TOD()                                                # object detector model
        if mode is 2:
            with tf.Session(graph=detector.detection_graph) as sess:
                while True:
                    output_keys = check_keys()
                    screen = np.array(sct.grab(region))                         # grab screen and convert it into array
                    screen = cv2.cvtColor(screen, code=cv2.COLOR_RGB2GRAY, dstCn=3)      # convert image into gray image
                    screen = np.stack((screen,)*3, axis=-1)
                    image_np = screen[np.newaxis, ...]

                    figure_vector = detector.detect(sess) + output_keys         # create vector of features and key
                    training_data.append(figure_vector)                         # add this new record into training_data
                    if len(training_data) % 10 == 0:
                        print('.', end='')
                    if len(training_data) % 100 == 0:
                        print('Training data at record -', len(training_data))
                        np.save(PATH_TO_TRAINING_FILE, training_data)           # save training data into xx.npy file
                    if keyboard.is_pressed('`'):
                        print("Stopping.")
                        if SHOW_DISPLAY is True:
                            cv2.destroyAllWindows()
                        break
        elif mode is 1:
            i = 0
            last_time = time.time()
            while True:
                i += 1
                print('Frame took {} seconds'.format(time.time() - last_time))
                last_time = time.time()
                screen = np.array(sct.grab(region))  # grab screen and convert it into array
                screen = screen[:, :, :3]
                filename = os.path.join(IMAGE_DIR, 'sct' + str(i) + '.png')
                cv2.imwrite(filename, screen, [cv2.IMWRITE_PNG_COMPRESSION, 8])
                if keyboard.is_pressed('`'):
                    print("Stopping. printed", i, "pictures")
                    break
