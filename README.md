### Mohammad Amaan's Submission

The submission contains following files :

1) Covid_Mask_Classifier.ipynb : Contains code to develop (build and train) a classifier that takes input an image of human face and classifies if the human has put on a mask.

Note : Additional data has been used in training so that the model experiences images from different kinds of distributions. Dataset details have been mentioned in references.

2) FaceDetector(FRRCNN).ipynb : Contains code to develop (build and train) a Faster RCNN face detector model.

3) Inference.ipynb : Contains code to integrate the above two models and use them to detect and classify faces with/without masks from images

4) classifier.pth : Saved model from Covid_Mask_Classifier.ipynb that is used in Inference.ipynb to classify faces.

5) customtrained_fasterrcnn_resnet50_fpn.pth : Saved model from FaceDetector(FRRCNN).ipynb that detects faces with/without mask in an image.

6) README.MD : Submission details.

What's new in this submission?

In this submission, i have quantized both the classifier.pth and customtrained_fasterrcnn_resnet_fpn.pth into the following files

7) quant_classifier.pth : Quantized face classifier. It is smaller than classifier.pth by **43%** and considerably faster in execution.

8) quant_detector.pth : Quantized face detector model. It is of same size as customtrained_fasterrcnn_resnet50_fpn.pht but it works on integer data type hence supports mobile and other IOT platforms. Also it gives considerable speed up in execution.


### Environments

I had to switch between these Environments to manage GPUs sessions.

Kaggle :
*  Covid_Mask_Classifier.ipynb
*  FaceDetector(FRRCNN).ipynb

Google Colab:
*  Inference.ipynb - **Run this file to analyse performance of my Submission**

Local System or Colab or Kaggle :
*  quantization.py

### References :

1) Notebooks from <a href="https://www.kaggle.com/c/global-wheat-detection">Global Wheat Detection challenge</a> on Kaggle .

2) <a href="https://www.kaggle.com/vtech6/medical-masks-dataset">Medical Face Masks</a> dataset and kernels on Kaggle .

3) <a href=https://pytorch.org/blog/introduction-to-quantization-on-pytorch/> Introduction to Quantization on PyTorch </a>
