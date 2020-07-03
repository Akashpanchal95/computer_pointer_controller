"""
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.

How to run :
python face_detection.py

"""

import os
import time
import cv2
import math

from openvino.inference_engine import IENetwork, IECore


class GazeEstimation:
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

        # Load model in network
        start_time = time.time()
        self.exec_net = self.plugin.load_network(network=self.net, device_name=self.device, num_requests=1)

        # Obtain blob info from network
        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))

        end_time = time.time()
        print(f"Gaze Estimation Model Loading Time: {end_time - start_time}")

    def predict(self, left_eye, right_eye, head_pose):
        """
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        """
        processed_left_eye, processed_right_eye = self.preprocess_input(left_eye, right_eye)
        # infer image
        outputs = self.exec_net.infer({'left_eye_image': processed_left_eye,
                                       'right_eye_image': processed_right_eye,
                                       'head_pose_angles': head_pose})

        (mouse_x, mouse_y), gaze_vector = self.preprocess_output(outputs, head_pose)

        return (mouse_x, mouse_y), gaze_vector

    def preprocess_input(self, left_eye, right_eye):
        """
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        """
        try:
            left_eye = cv2.resize(left_eye, (60, 60))
            right_eye = cv2.resize(right_eye, (60, 60))

            preprocessed_left_image = left_eye.transpose((2, 0, 1))
            preprocessed_right_image = right_eye.transpose((2, 0, 1))

            preprocessed_left_image = preprocessed_left_image.reshape(1, 3, 60, 60)
            preprocessed_right_image = preprocessed_right_image.reshape(1, 3, 60, 60)
        except:
            # print("Left Eye",left_eye.shape)
            # print("Right Eye",right_eye.shape)
            return 0, 0

        return preprocessed_left_image, preprocessed_right_image

    def preprocess_output(self, outputs, head_pose):
        """
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        """
        gaze_vector = outputs[self.out_blob][0]

        # HeadPoseEstimation model find angle_r_fc output
        rollValue = head_pose[2]
        cosValue = math.cos(rollValue * math.pi / 180.0)
        sinValue = math.sin(rollValue * math.pi / 180.0)

        mouse_x = gaze_vector[0] * cosValue + gaze_vector[1] * sinValue
        mouse_y = -gaze_vector[0] * sinValue + gaze_vector[1] * cosValue

        # gaze_vec = {'x': outputs['gaze_vector'][0][0].item(),
        #             'y': outputs['gaze_vector'][0][1].item(),
        #             'z': outputs['gaze_vector'][0][2].item()}

        return (mouse_x, mouse_y), gaze_vector


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Face detection and save crop image')

    parser.add_argument('--input',
                        default='../bin/das.jpg',
                        # default='../output/face_detection.jpg',

                        type=str,
                        help='open image file and detect face')

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

    parser.add_argument('--head_pose',
                        default='../models/intel/head-pose-estimation-adas-0001/'
                                'FP32/head-pose-estimation-adas-0001.xml',
                        type=str,
                        help='path to the face detection model')

    parser.add_argument('--gaze_estimation',
                        default='../models/intel/gaze-estimation-adas-0002/'
                                'FP32/gaze-estimation-adas-0002.xml',
                        type=str,
                        help='path to the face detection model')

    parser.add_argument('--device',
                        default='CPU',
                        type=str,
                        choices=['CPU', 'GPU', 'FPGA', 'MYRIAD'],
                        help='Choose device to run inference from CPU, GPU, MYRIAD, FPGA')

    parser.add_argument('--threshold',
                        default=0.5,
                        type=int)

    parser.add_argument('--output',
                        default='output',
                        type=str,
                        help='output directory to save results')

    args = parser.parse_args()

    # Read Image from given input
    image = cv2.imread(args.input)

    from head_pose_estimation import HeadPoseEstimation
    from facial_landmarks_detection import FaceLandmark
    from face_detection import FaceDetection

    FD = FaceDetection(args.face_detection, args.device)
    FD.load_model()
    crop_face, coords = FD.predict(image, visualize=True)

    FL = FaceLandmark(args.face_landmark, args.device)
    FL.load_model()
    left_eye, right_eye, left_eye_crop, right_eye_crop = FL.predict(crop_face, visualize=True)

    HE = HeadPoseEstimation(args.heas_pose, args.device)
    HE.load_model()
    HPE = HE.predict(crop_face, image)

    GE = GazeEstimation(args.gaze_estimation, args.device)
    GE.load_model()

    (newx, newy), gaze_vector = GE.predict(left_eye_crop, right_eye_crop, HPE)

    (left_eye_gaze) = int(left_eye[0] + gaze_vector[0] * 100), int(left_eye[1] - gaze_vector[1] * 100)
    # newx_l, newy_l,
    (right_eye_gaze) = int(right_eye[0] + gaze_vector[0] * 100), int(right_eye[1] - gaze_vector[1] * 100)
    # newx_r, newy_r

    frame = cv2.arrowedLine(crop_face, (right_eye), (left_eye_gaze), (0, 0, 255), 2)
    frame = cv2.arrowedLine(crop_face, (left_eye), (right_eye_gaze), (255, 0, 0), 2)

    cv2.imshow("gaze", image)
    cv2.waitKey(0)
