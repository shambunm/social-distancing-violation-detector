# social-distancing-violation-detector
Social Distancing detector using Darknet, OpenCv- DNN.
YOLOv3 uses Darknet-53. Darknet-53 is a backbone feature extractor, it has 53 convolutional layers.
From coco dataset label file is loaded and index of person is identified.
Video is loaded and size changed to suit yolo input.
Bounding boxes and centroids detected, euclidean distance of centroid calculated.
Color of bounding boxes defined.
If distance less than threshold defined then display social distancing violated.
Save the output video to disk.
