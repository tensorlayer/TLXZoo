<a href="https://tensorlayerx.readthedocs.io/">
    <div align="center">
        <img src="https://git.openi.org.cn/hanjr/tensorlayerx-image/raw/branch/master/tlx-LOGO-04.png" width="50%" height="30%"/>
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
