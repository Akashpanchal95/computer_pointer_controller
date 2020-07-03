# Computer Pointer Controller

Computer Pointer Controller app is able to control the mouse pointer using eye location and head pose moment for more information please read the project flow.
   
#### Project Flow
* Step1: Detect face and crop from image.
* Step2: Detect face landmark from the cropped image.
* Step3: Find and crop the left & right eye from the cropped image.
* Step4: Find the Head pose from the cropped image.
* Step5: Combine the cropped eyes and head pose to estimate the gaze of eye direction. 
 
## Project Set Up and Installation
Prerequisites:
* OpenVINO 2020.x , [Click here](https://docs.openvinotoolkit.org/latest/index.html) to download and setup.
* Install requirement using ```pip install -r requirement.txt```
* Download the Models which required to run this project. 
* Required Models (Click for more details): 
    * [face-detection-adas-binary-0001](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html) 
    * [landmarks-regression-retail-0009](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
    * [gaze-estimation-adas-0002](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
    * [head-pose-estimation-adas-0001](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)
    
Below command download the folder inside the intel folder. you can find in below directory structure.
```
python <openvino-installation-path>/deployment_tools/tools/model_downloader/downloader.py --name face-detection-adas-0001 --output_dir models
```

### Directory Structure
```
├── bin
│   ├── demo.mp4
├── models
│   └── intel
│       ├── face-detection-adas-binary-0001
│       │   └── FP32-INT1
│       │       ├── face-detection-adas-binary-0001.bin
│       │       └── face-detection-adas-binary-0001.xml
│       ├── gaze-estimation-adas-0002
│       │   ├── FP16
│       │   │   ├── gaze-estimation-adas-0002.bin
│       │   │   └── gaze-estimation-adas-0002.xml
│       │   ├── FP16-INT8
│       │   │   ├── gaze-estimation-adas-0002.bin
│       │   │   └── gaze-estimation-adas-0002.xml
│       │   └── FP32
│       │       ├── gaze-estimation-adas-0002.bin
│       │       └── gaze-estimation-adas-0002.xml
│       ├── head-pose-estimation-adas-0001
│       │   ├── FP16
│       │   │   ├── head-pose-estimation-adas-0001.bin
│       │   │   └── head-pose-estimation-adas-0001.xml
│       │   ├── FP16-INT8
│       │   │   ├── head-pose-estimation-adas-0001.bin
│       │   │   └── head-pose-estimation-adas-0001.xml
│       │   └── FP32
│       │       ├── head-pose-estimation-adas-0001.bin
│       │       └── head-pose-estimation-adas-0001.xml
│       └── landmarks-regression-retail-0009
│           ├── FP16
│           │   ├── landmarks-regression-retail-0009.bin
│           │   └── landmarks-regression-retail-0009.xml
│           ├── FP16-INT8
│           │   ├── landmarks-regression-retail-0009.bin
│           │   └── landmarks-regression-retail-0009.xml
│           └── FP32
│               ├── landmarks-regression-retail-0009.bin
│               └── landmarks-regression-retail-0009.xml
├── output
│   ├── demo_output_final.mp4
│   ├── demo_output.mp4
│   ├── face_detection1.jpg
│   └── face_detection.jpg
├── README.md
├── requirements.txt

```

## Demo
```
python src/main.py
```

## Documentation
Main file arguments:

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
                                'FP32/landmarks-regression-retail-0009.xml',
                        type=str,
                        help='path to the face landmark model')

    parser.add_argument('--head_pose',
                        default='../models/intel/head-pose-estimation-adas-0001/'
                                'FP32/head-pose-estimation-adas-0001.xml',
                        type=str,
                        help='path to the head pose model')

    parser.add_argument('--gaze_estimation',
                        default='../models/intel/gaze-estimation-adas-0002/'
                                'FP32/gaze-estimation-adas-0002.xml',
                        type=str,
                        help='path to the gaze estimation model')

    parser.add_argument('--visualize',
                        default=True,
                        action='store_true',
                        help='individual visualize the model output')

    parser.add_argument('--output_dir',
                        default='../output',
                        type=str,
                        help='output directory to save video results')


## Benchmarks
####Model Loading Time ON CPU & GPU
Processor Details: Intel(R) Core(TM) i5-7400 CPU @ 3.00GHz

<p align="center">
<img src="resources/fp16_32_cpu.png" width=400px height=350px>
</p>
<br>
<p align="center">
<img src="resources/fp16_32_gpu.png" width=400px height=350px>
</p>


## Results
As we can see on the graph, when we run the model on GPU it takes most of time in model loading.

Regarding accuracy, it wasn't find any big differences between FP16 and FP32 models. This could be relevant because with FP16 models we can use hardware accelerators like VPUs without worrying about accuracy loss.

### Edge Cases
When multiple people are in the frame then the application work with only one person. This works good but sometime it may show flickering effect between two detected heads.

At a time only extract the one face and inference on it, other faces are ignored. 

So as per the above edge cases, we need to make sure that there are enough lighting and only a single person in the frame.