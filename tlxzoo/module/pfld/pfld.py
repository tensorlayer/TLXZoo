import tensorlayerx as tlx
from tensorlayerx import nn


def pfld_loss(landmarks, angle, landmark_gt, euler_angle_gt, attribute_gt):
    batchsize = tlx.get_tensor_shape(landmarks)[0]

    weight_angle = tlx.reduce_sum(1 - tlx.cos(angle - euler_angle_gt), axis=1)

    if attribute_gt:
        attributes_w_n = tlx.cast(attribute_gt, tlx.float32)
        mat_ratio = tlx.reduce_mean(attributes_w_n, axis=0)
        mat_ratio = tlx.convert_to_tensor(
            [1.0 / (x) if x > 0 else batchsize for x in mat_ratio], dtype=tlx.float32)
        weight_attribute = tlx.reduce_sum(
            tlx.multiply(attributes_w_n, mat_ratio), axis=1)
    else:
        weight_attribute = 1

    l2_distant = tlx.reduce_sum(
        (landmark_gt - landmarks) * (landmark_gt - landmarks), axis=1)

    return tlx.reduce_mean(weight_angle * weight_attribute * l2_distant)


def conv_bn(oup, kernel, stride, padding="SAME"):
    return nn.Sequential(
        nn.Conv2d(oup, kernel, stride, padding=padding, b_init=None),
        nn.BatchNorm2d(),
        nn.ReLU()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = use_res_connect

        self.conv = nn.Sequential(
            nn.Conv2d(inp * expand_ratio, (1, 1), (1, 1),
                      padding="VALID", b_init=None),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.GroupConv2d(
                inp * expand_ratio,
                (3, 3),
                (stride, stride),
                inp * expand_ratio,
                padding="SAME",
                b_init=None
            ),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.Conv2d(oup, (1, 1), (1, 1), padding="VALID", b_init=None),
            nn.BatchNorm2d(),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class PFLDBackbone(nn.Module):
    def __init__(self):
        super(PFLDBackbone, self).__init__()

        self.conv1 = nn.Conv2d(
            64,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding="SAME",
            b_init=None
        )
        self.bn1 = nn.BatchNorm2d()
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding="SAME",
            b_init=None
        )
        self.bn2 = nn.BatchNorm2d()

        self.conv3_1 = InvertedResidual(64, 64, 2, False, 2)
        self.block3_2 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_3 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_4 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_5 = InvertedResidual(64, 64, 1, True, 2)

        self.conv4_1 = InvertedResidual(64, 128, 2, False, 2)

        self.conv5_1 = InvertedResidual(128, 128, 1, False, 4)
        self.block5_2 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_3 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_4 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_5 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_6 = InvertedResidual(128, 128, 1, True, 4)

        self.conv6_1 = InvertedResidual(128, 16, 1, False, 2)  # [16, 14, 14]

        self.conv7 = conv_bn(32, (3, 3), (2, 2))  # [32, 7, 7]
        self.conv8 = nn.Conv2d(128, (7, 7), (1, 1),
                               padding="VALID")  # [128, 1, 1]
        self.bn8 = nn.BatchNorm2d()

        # self.avg_pool1 = nn.MeanPool2d((14, 14), (14, 14))
        # self.avg_pool2 = nn.MeanPool2d((7, 7), (7, 7))
        self.fc = nn.Linear(136)

        self.build((1, 112, 112, 3))

    def build(self, inputs_shape):
        ones = tlx.ones(inputs_shape)
        _ = self(ones)

    def forward(self, x):  # x: 3, 112, 112
        x = self.relu(self.bn1(self.conv1(x)))  # [64, 56, 56]
        x = self.relu(self.bn2(self.conv2(x)))  # [64, 56, 56]
        x = self.conv3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        features = self.block3_5(x)

        x = self.conv4_1(features)
        x = self.conv5_1(x)
        x = self.block5_2(x)
        x = self.block5_3(x)
        x = self.block5_4(x)
        x = self.block5_5(x)
        x = self.block5_6(x)
        x = self.conv6_1(x)
        # x1 = self.avg_pool1(x)
        x1 = tlx.reshape(x, (tlx.get_tensor_shape(x)[0], -1))

        x = self.conv7(x)
        # x2 = self.avg_pool2(x)
        x2 = tlx.reshape(x, (tlx.get_tensor_shape(x)[0], -1))

        x = self.relu(self.conv8(x))
        x3 = tlx.reshape(x, (tlx.get_tensor_shape(x)[0], -1))

        multi_scale = tlx.concat([x1, x2, x3], 1)
        landmarks = self.fc(multi_scale)
        return landmarks, features


class AuxiliaryNet(nn.Module):
    def __init__(self):
        super(AuxiliaryNet, self).__init__()
        self.conv1 = conv_bn(128, (3, 3), (2, 2))
        self.conv2 = conv_bn(128, (3, 3), (1, 1))
        self.conv3 = conv_bn(32, (3, 3), (2, 2))
        self.conv4 = conv_bn(128, (7, 7), (1, 1), padding='VALID')
        # self.max_pool1 = nn.MaxPool2d((3, 3), (3, 3))
        self.fc1 = nn.Linear(32)
        self.fc2 = nn.Linear(3)

        self.build((1, 28, 28, 64))

    def build(self, inputs_shape):
        ones = tlx.ones(inputs_shape)
        _ = self(ones)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.max_pool1(x)
        x = tlx.reshape(x, (tlx.get_tensor_shape(x)[0], -1))
        x = self.fc1(x)
        x = self.fc2(x)

        return x


class PFLD(nn.Module):
    def __init__(self, **kwargs):
        super(PFLD, self).__init__()
        self.backbone = PFLDBackbone()
        self.auxiliarynet = AuxiliaryNet()

    def forward(self, x):
        return self.backbone(x)

    def loss_fn(self, output, target):
        landmarks, features = output
        angle = self.auxiliarynet(features)

        if len(target) == 3:
            return pfld_loss(landmarks, angle, target[0], target[1], target[2])
        else:
            return pfld_loss(landmarks, angle, target[0], target[1], None)
