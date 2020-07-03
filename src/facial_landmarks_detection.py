"""
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.

How to run :
Without viewing result
python facial_landmarks_detection.py

With Show result
python facial_landmarks_detection.py --visualize True

"""

import os
import time
import cv2
import numpy as np
import logging

from openvino.inference_engine import IENetwork, IECore


class FaceLandmark:
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

        print(f"Face Landmark Model Loading Time: {end_time - start_time}")
        logging.info(f"Face Landmark Model Loading Time: {end_time - start_time}")

    def predict(self, image, visualize=False):
        """
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        """
        preprocessed_image = self.preprocess_input(image)
        # infer image
        outputs = self.exec_net.infer({self.input_blob: preprocessed_image})

        coords = self.preprocess_output(outputs)
        if len(coords) == 0:
            logging.error("No coordinate found from face")
            return 0, 0, 0, 0

        h = image.shape[0]
        w = image.shape[1]

        coords = coords * np.array([w, h, w, h])
        coords = coords.astype(np.int32)

        left_eye = (coords[0], coords[1])
        right_eye = (coords[2], coords[3])

        # create box from center point, 4 side expand ratio
        expand_eye_size_x = w * 0.20
        expand_eye_size_y = h * 0.20

        right_eye_dimensions = [int(right_eye[1] - expand_eye_size_y / 2),
                                int(right_eye[1] + expand_eye_size_y / 2),
                                int(right_eye[0] - expand_eye_size_x / 2),
                                int(right_eye[0] + expand_eye_size_x / 2)]

        left_eye_dimensions = [int(left_eye[1] - expand_eye_size_y / 2),
                               int(left_eye[1] + expand_eye_size_y / 2),
                               int(left_eye[0] - expand_eye_size_x / 2),
                               int(left_eye[0] + expand_eye_size_x / 2)]

        right_eye_crop = image[right_eye_dimensions[0]:right_eye_dimensions[1],
                         right_eye_dimensions[2]:right_eye_dimensions[3]]

        # print(right_eye_dimensions[0], right_eye_dimensions[1],
        #       right_eye_dimensions[2], right_eye_dimensions[3])

        left_eye_crop = image[left_eye_dimensions[0]:left_eye_dimensions[1],
                        left_eye_dimensions[2]:left_eye_dimensions[3]]

        # print(left_eye_dimensions[0], left_eye_dimensions[1],
        #       left_eye_dimensions[2], left_eye_dimensions[3])

        image = cv2.circle(image, (coords[0], coords[1]), 3, 255, -1)
        image = cv2.circle(image, (coords[2], coords[3]), 3, 255, -1)

        cv2.rectangle(image, (right_eye_dimensions[2], right_eye_dimensions[0]),
                      (right_eye_dimensions[3], right_eye_dimensions[1]), (255, 255, 0))

        cv2.rectangle(image, (left_eye_dimensions[2], left_eye_dimensions[0]),
                      (left_eye_dimensions[3], left_eye_dimensions[1]), (255, 255, 0))

        if visualize:
            # Show the crop image,right_eye_crop,left_eye_crop
            # cv2.imshow("Right", right_eye_crop)
            # cv2.imshow("Left", left_eye_crop)
            cv2.imshow("Face Landmark", image)
            cv2.waitKey(0)

        return left_eye, right_eye, left_eye_crop, right_eye_crop

    def preprocess_input(self, image):
        """
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        """
        preprocessed_image = cv2.resize(image, (48, 48))
        preprocessed_image = preprocessed_image.transpose((2, 0, 1))
        preprocessed_image = preprocessed_image.reshape(1, 3, 48, 48)

        return preprocessed_image

    def preprocess_output(self, outputs):
        """
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.

            The net outputs a blob with the shape: [1, 10], containing a row-vector of 10 floating point
            values for five landmarks coordinates in the form (x0, y0, x1, y1, ..., x5, y5).
            All the coordinates are normalized to be in range [0,1].
        """
        outs = outputs[self.out_blob][0]

        # Fetch eye co-ordinates from the model
        left_eye_x = outs[0].tolist()[0][0]
        left_eye_y = outs[1].tolist()[0][0]
        right_eye_x = outs[2].tolist()[0][0]
        right_eye_y = outs[3].tolist()[0][0]

        return left_eye_x, left_eye_y, right_eye_x, right_eye_y


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Face detection and save crop image')

    parser.add_argument('--input',
                        default='../bin/modi.jpg',
                        type=str,
                        help='path of input image')

    parser.add_argument('--face_landmark',
                        default='../models/intel/landmarks-regression-retail-0009/'
                                'FP32/landmarks-regression-retail-0009.xml',
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


    FD = FaceDetection(args.face_detection, args.device)
    FD.load_model()
    crop_face, coords = FD.predict(image, visualize=args.visualize)

    FL = FaceLandmark(args.face_landmark, args.device)
    FL.load_model()
    image = FL.predict(crop_face, visualize=args.visualize)
