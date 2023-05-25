<a href="https://tensorlayerx.readthedocs.io/">
    <div align="center">
        <img src="https://git.openi.org.cn/hanjr/tensorlayerx-image/raw/branch/master/tlx-LOGO--02.jpg" width="60%" height="50%"/>
    </div>
</a>

Pre-trained models based on TensorLayerX. 
TensorLayerX is a multi-backend AI framework, which can run on almost all operation systems and AI hardwares, and support hybrid-framework programming. The currently version supports TensorFlow, MindSpore, PaddlePaddle, PyTorch, OneFlow and Jittor as the backends.

# Quick Start
## Installation
### pip
```bash
# install from pypi
pip3 install tlxzoo
```

### docker
```bash
docker pull xiaolong0612/tlxzoo:0.0.1
# run docker
cd /workspace/tlxzoo
python demo/vision/image_classification/vgg/predict.py
```

## train
```bash
python demo/vision/image_classification/vgg/train.py
```

### pretrained model
https://pan.baidu.com/s/1FZiHSthgX2FqynBP9cJ6_A?pwd=8dbs

## predict

```bash
python demo/vision/image_classification/vgg/predict.py
```

# Scenes and Models

TLXZOO currently covers 4 fields and 12 tasks. They ares listed below:

## Vision

### Object Detection
- [DETR](demo/vision/object_detection/detr)
- [PPYOLOE](demo/vision/object_detection/ppyoloe)

### Image Classification
- [ResNet](demo/vision/image_classification/resnet)
- [VGG](demo/vision/image_classification/vgg)
- [EfficientNet](demo/vision/image_classification/efficientnet)

### Image Segmentation
- [UNet](demo/vision/image_segmentation/unet)

### Face Recognition
- [RetinaFace](demo/vision/face_recognition/retinaface)
- [ArcFace](demo/vision/face_recognition/arcface)

### Human Pose Estimation
- [HRNet](demo/vision/human_pose_estimation/hrnet)

### Facial Landmark Detection
- [PFLD](demo/vision/facial_landmark_detection/pfld)

### OCR
- [TrOCR](demo/vision/ocr/trocr)

## Text

### Text Classification
- [BERT](demo/text/text_classification/bert)
- [T5](demo/text/text_classification/t5)

### Text Conditional Generation
- [T5](demo/text/nmt/t5)
  
### Text Token Classidication
- [BERT](demo/text/token_classification/bert)
- [T5](demo/text/token_classification/t5)

## Speech

### Automatic Speech Recognition
- [wav2vec](demo/speech/automatic_speech_recognition/wav2vec)

## Graph

### Node Classification
- [GCN](tlxzoo/module/gcn/gcn.py)