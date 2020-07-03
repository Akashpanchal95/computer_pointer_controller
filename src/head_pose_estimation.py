"""
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.

How to run :
Without viewing result
python head_pose_estimation.py

With Show result
python head_pose_estimation.py --visualize True
"""

import os
import time
import logging
from math import cos, sin

import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore


class HeadPoseEstimation:
    """
    Class for the Face Detection Model.
    """

    def __init__(self, model_name, device='CPU', extensions=None):
        self.net = None
        self.plugin = None
        self.input_blob = None
        self.out_blob = None
        self.exec_net = None
        self.model_name = model_name
        self.extensions = extensions
        self.device = device

    def load_model(self):
        """
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        """

        # Fetch XML model
        model_xml = self.model_name
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        self.plugin = IECore()
        self.net = IENetwork(model=model_xml, weights=model_bin)

        # Add CPU extension to self.plugin and check not supported layers
        if "CPU" in self.device:
            supported_layers = self.plugin.query_network(self.net, self.device)
            not_supported_layers = [layer for layer in self.net.layers.keys() if layer not in supported_layers]

            if len(not_supported_layers) != 0 and self.device == 'CPU':
                print(f"Not supported layers: {not_supported_layers}")
                logging.error(f"Unsupported layers: {not_supported_layers}")

        # Load model in network
        start_time = time.time()
        self.exec_net = self.plugin.load_network(network=self.net, device_name=self.device, num_requests=1)
        end_time = time.time()

        # Obtain blob info from network
        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))

        print(f"Head Pose Model Loading Time: {end_time - start_time}")
        logging.info(f"Head Pose Model Loading Time: {end_time - start_time}")

    def predict(self, image, visualize=False):
        """
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        """
        preprocessed_image = self.preprocess_input(image)
        # infer image
        outputs = self.exec_net.infer({self.input_blob: preprocessed_image})

        angle_outputs = self.preprocess_output(outputs)
        size = 100
        height, width = image.shape[:2]

        tdx = width / 2
        tdy = height / 2

        yaw = angle_outputs[0] * np.pi / 180.0
        pitch = angle_outputs[1] * np.pi / 180.0
        roll = angle_outputs[2] * np.pi / 180.0

        # X-Axis pointing to right. drawn in red
        x1 = size * (cos(yaw) * cos(roll)) + tdx
        y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

        # Y-Axis | drawn in green
        x2 = size * (-cos(yaw) * sin(roll)) + tdx
        y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

        # Z-Axis (out of the screen) drawn in blue
        x3 = size * (sin(yaw)) + tdx
        y3 = size * (-cos(yaw) * sin(pitch)) + tdy

        cv2.line(image, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
        cv2.line(image, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
        cv2.line(image, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

        if visualize:
            cv2.imshow("HEAD POSE", image)
            cv2.waitKey(0)

        return angle_outputs

    def preprocess_input(self, image):
        """
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        """
        preprocessed_image = cv2.resize(image, (60, 60))
        preprocessed_image = preprocessed_image.transpose((2, 0, 1))
        preprocessed_image = preprocessed_image.reshape(1, 3, 60, 60)

        return preprocessed_image

    def preprocess_output(self, outputs):
        """
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        """
        angle_output = [outputs['angle_y_fc'].tolist()[0][0],
                        outputs['angle_p_fc'].tolist()[0][0],
                        outputs['angle_r_fc'].tolist()[0][0]]

        return angle_output


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Face detection and save crop image')

    parser.add_argument('--input',
                        default='../bin/modi.jpg',
                        type=str,
                        help='open image file and detect face')

    parser.add_argument('--head_pose',
                        default='../models/intel/head-pose-estimation-adas-0001/'
                                'FP32/head-pose-estimation-adas-0001.xml',
                        type=str,
                        help='path to the face detection model')

    parser.add_argument('--face_detection',
                        default='../models/intel/face-detection-adas-binary-0001/'
                                'FP32-INT1/face-detection-adas-binary-0001.xml',
                        type=str,
                        help='path to the face detection model')

    parser.add_argument('--device',
                        default='CPU',
                        type=str,
                        choices=['CPU', 'GPU', 'FPGA', 'MYRIAD'],
                        help='Choose device to run inference from CPU, GPU, MYRIAD, FPGA')

    parser.add_argument('--visualize',
                        default=False,
                        type=bool,
                        help='visualize the model output')

    parser.add_argument('--output',
                        default='output',
                        type=str,
                        help='output directory to save results')

    args = parser.parse_args()

    from face_detection import FaceDetection

    # Read Image from given input
    image = cv2.imread(args.input)
    logging.info(f"Reading Image {args.input}")

    FD = FaceDetection(args.face_detection, args.device)
    FD.load_model()
    crop_face, coords = FD.predict(image, visualize=args.visualize)

    if coords is 0:
        logging.warning(f"Face not found from image :{args.input}")
        exit()

    HPE = HeadPoseEstimation(args.head_pose, args.device)
    HPE.load_model()
    image = HPE.predict(crop_face, args.visualize)
