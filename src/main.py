"""
__Author__: Akash Panchal
"""
import argparse
import logging

import cv2
from face_detection import FaceDetection
from facial_landmarks_detection import FaceLandmark
from gaze_estimation import GazeEstimation
from head_pose_estimation import HeadPoseEstimation
from input_feeder import InputFeeder
from mouse_controller import MouseController


def parse_args():
    parser = argparse.ArgumentParser(description='Control your mouse position using computer vision with your eye gaze')
    parser.add_argument('--input',
                        default='../bin/demo.mp4',
                        type=str,
                        help='open video file or image file')

    parser.add_argument('--device',
                        default='CPU',
                        type=str,
                        choices=['CPU', 'GPU', 'MYRIAD', 'FPGA'],
                        help='Run inference on, one of CPU, GPU, MYRIAD, FPGA')

    parser.add_argument('--face_detection',
                        default='../models/intel/face-detection-adas-binary-0001/'
                                'FP32-INT1/face-detection-adas-binary-0001.xml',
                        type=str,
                        help='path to the face detection model')

    parser.add_argument('--face_landmark',
                        default='../models/intel/landmarks-regression-retail-0009/'
                                'FP16-INT8/landmarks-regression-retail-0009.xml',
                        type=str,
                        help='path to the face landmark model')

    parser.add_argument('--head_pose',
                        default='../models/intel/head-pose-estimation-adas-0001/'
                                'FP16-INT8/head-pose-estimation-adas-0001.xml',
                        type=str,
                        help='path to the head pose model')

    parser.add_argument('--gaze_estimation',
                        default='../models/intel/gaze-estimation-adas-0002/'
                                'FP16-INT8/gaze-estimation-adas-0002.xml',
                        type=str,
                        help='path to the gaze estimation model')

    parser.add_argument('--show_result',
                        default=True,
                        type=bool,
                        help='visualize the model output')

    parser.add_argument('--visualize',
                        default=False,
                        type=bool,
                        help='visualize the model output')

    parser.add_argument('--output_dir',
                        default='../output',
                        type=str,
                        help='output directory to save video results')

    parser.add_argument('--log-level',
                        default='error',
                        type=str,
                        choices=['debug', 'info', 'warning', 'error'],
                        help='the log level, one of debug, info, warning, error, critical')

    args = parser.parse_args()
    return args


def main(args):
    # set log level
    levels = {'debug': logging.DEBUG,
              'info': logging.INFO,
              'warning': logging.WARNING,
              'error': logging.ERROR}

    log_level = levels.get(args.log_level, logging.ERROR)

    logging.basicConfig(level=log_level)

    mouse_control = MouseController('high', 'fast')

    logging.info("Model Loading Please Wait ..")
    face_det = FaceDetection(args.face_detection, args.device)
    facial_det = FaceLandmark(args.face_landmark, args.device)
    head_pose_est = HeadPoseEstimation(args.head_pose, args.device)
    gaze_est = GazeEstimation(args.gaze_estimation, args.device)
    logging.info("Model loading successfully")

    inp = InputFeeder(input_type='video', input_file=args.input)
    inp.load_data()

    face_det.load_model()
    facial_det.load_model()
    head_pose_est.load_model()
    gaze_est.load_model()

    video_writer = cv2.VideoWriter(args.output_dir + '/demo_output11.mp4', cv2.VideoWriter_fourcc(*'MPEG'), 15,
                                   (1920, 1080), True)

    cv2.namedWindow('gaze')
    for frame in inp.next_batch():
        try:
            frame.shape
        except Exception as err:
            break
        crop_face, crop_coords = face_det.predict(frame, visualize=args.visualize)

        left_eye, right_eye, left_eye_crop, right_eye_crop = facial_det.predict(crop_face, visualize=args.visualize)
        head_pose = head_pose_est.predict(crop_face, visualize=args.visualize)

        (new_x, new_y), gaze_vector = gaze_est.predict(left_eye_crop, right_eye_crop, head_pose)

        left_eye_gaze = int(left_eye[0] + gaze_vector[0] * 100), int(left_eye[1] - gaze_vector[1] * 100)
        right_eye_gaze = int(right_eye[0] + gaze_vector[0] * 100), int(right_eye[1] - gaze_vector[1] * 100)

        cv2.arrowedLine(crop_face, left_eye, left_eye_gaze, (0, 0, 255), 2)
        cv2.arrowedLine(crop_face, right_eye, right_eye_gaze, (0, 0, 255), 2)

        video_writer.write(frame)
        mouse_control.move(new_x, new_y)

        if args.show_result:
            cv2.imshow('gaze', frame)
            cv2.waitKey(1)

    inp.close()
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()
    main(args)
