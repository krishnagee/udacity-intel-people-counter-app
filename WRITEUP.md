# Project Write-Up

The counter uses the Inference Engine included in the Intel® Distribution of OpenVINO™ Toolkit. The model used is be able to identify people in a video frame. The application counts the number of people in the current frame, the duration that a person is in the frame (time elapsed between entering and exiting a frame) and the total count of people. It then sends the data to a local web server using the Paho MQTT Python package.

Hardware Requirements
6th to 10th generation Intel® Core™ processor with Iris® Pro graphics or Intel® HD Graphics.
OR use of Intel® Neural Compute Stick 2 (NCS2)

Software Requirements
Intel® Distribution of OpenVINO™ toolkit 2019 R3 release
Node v6.17.1
Npm v3.10.10
CMake
MQTT Mosca server
Python 3.5 or 3.6

## Explaining Custom Layers

Ssd_inception_v2_coco and faster_rcnn_inception_v2_coco performed good as compared to rest of the models, but, in this project, faster_rcnn_inception_v2_coco is used which is fast in detecting people with less errors.

Intel openVINO already contains extensions for custom layers used in TensorFlow Object Detection Model Zoo.

Downloading the model from the GitHub repository of Tensorflow Object Detection Model Zoo by the following command:

wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

Extracting the tar.gz file by the following command:

tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

Changing the directory to the extracted folder of the downloaded model:

cd faster_rcnn_inception_v2_coco_2018_01_28

The model can't be the existing models provided by Intel. So, converting the TensorFlow model to Intermediate Representation (IR) or OpenVINO IR format. The command used is given below:

python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json

## Comparing Model Performance

For this project, various classes of models were tested from the TensorFlow Object Detection Model Zoo. SSD_inception_v2_coco and faster_rcnn_inception_v2_coco performed good as compared to rest of the models, but, in this project, faster_rcnn_inception_v2_coco is used which is fast in detecting people with less errors. Intel openVINO already contains extensions for custom layers used in TensorFlow Object Detection Model Zoo.


## Assess Model Use Cases

This application could keep a check on the number of people in a particular area and could be helpful where there is restriction on the number of people present in a particular area. Further, with some updations, this could also prove helpful in the current COVID-19 scenario i.e. to keep a check on the number of people in the frame. Also, model could also be used as part of survellience and security in government/military restricted area to detect the entry of people and record their entry and exit timings with their identity.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model.Various insights could be drawn on the model by testing it with different videos and analyzing the model performance on low light input videos. This would be an important factor in determining the best model for the given scenario.

## Model Research

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Faster R-CNN Inception V2 COCO]
[http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz]

python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json

  
- Model 2: [SSD Inception V2 COCO]
[http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz]

python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json

- Model 3: [person-detection-retail-0013]
[https://docs.openvinotoolkit.org/2019_R1/_person_detection_retail_0013_description_person_detection_retail_0013.html]

The intermediate representation of the model is available in the intel model repository. 

