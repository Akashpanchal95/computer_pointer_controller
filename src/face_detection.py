"""
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.

How to run :
Without viewing result
python face_detection.py

With Show result
python face_detection.py --visualize Tru

"""

import os
import time
import cv2
import numpy as np
import logging

from openvino.inference_engine import IENetwork, IECore


class FaceDetection:
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
                logging.error(f"Unsupported layers: {not_supported_layers}")
                print(f"Not supported layers: {not_supported_layers}")

        # Load model in network
        start_time = time.time()
        self.exec_net = self.plugin.load_network(network=self.net, device_name=self.device, num_requests=1)
        end_time = time.time()

        # Obtain blob info from network
        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))

        print(f"Face Detection Model Loading Time: {end_time - start_time}")
        logging.info(f"Face Detection Model Loading Time: {end_time - start_time}")

    def predict(self, image, visualize):
        """
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        """
        # org_img = image.copy
        preprocessed_image = self.preprocess_input(image)
        # infer image
        outputs = self.exec_net.infer({self.input_blob: preprocessed_image})

        coords = self.preprocess_output(outputs)

        if len(coords) == 0:
            logging.warning("No face found in video or image")
            return 0, 0

        coords = coords[0]  # take the first detected face
        height = image.shape[0]
        width = image.shape[1]
        coords = coords * np.array([width, height, width, height])
        coords = coords.astype(np.int32)

        cropped_face = image[coords[1]:coords[3], coords[0]:coords[2]]
        cv2.rectangle(image, (coords[0], coords[1]), (coords[2], coords[3]), (255, 12, 12), 2)

        if visualize:

            # Save Image
            # cv2.imwrite('../output/face_detection1.jpg', cropped_face)
            # cv2.rectangle(image, (coords[0], coords[1]), (coords[2], coords[3]), (255, 12, 12), 2)
            cv2.imshow("Face detected", image)
            cv2.waitKey(0)
        else:
            logging.info("Visualization is off so image is not visible")
        return cropped_face, coords

    def preprocess_input(self, image):
        """
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        """
        preprocessed_image = cv2.resize(image, (672, 384))
        preprocessed_image = preprocessed_image.transpose((2, 0, 1))
        preprocessed_image = preprocessed_image.reshape(1, 3, 384, 672)

        return preprocessed_image

    def preprocess_output(self, outputs):
        """
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        """
        coords = []
        outs = outputs[self.out_blob][0][0]
        logging.info(f"Total {len(outs)} face found")
        for out in outs:
            confidence = out[2]
            if confidence > 0.5:  # args.threshold:
                x_min = out[3]
                y_min = out[4]
                x_max = out[5]
                y_max = out[6]
                coords.append([x_min, y_min, x_max, y_max])
            logging.info(f"Face coordinate: {coords}")

        return coords


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Face detection and save crop image')

    parser.add_argument('--input',
                        default='../bin/modi.jpg',
                        type=str,
                        help='open image file and detect face')

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

    parser.add_argument('--threshold',
                        default=0.5,
                        type=int)

    parser.add_argument('--output',
                        default='output',
                        type=str,
                        help='output directory to save results')

    args = parser.parse_args()

    # Read Image from given input
    logging.info(f"Reading Image {args.input}")
    image = cv2.imread(args.input)
    FD = FaceDetection(args.face_detection, args.device)
    FD.load_model()
    crop_face, coords = FD.predict(image, args.visualize)
