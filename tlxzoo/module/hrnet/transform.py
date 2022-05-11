import numpy as np
from tensorlayerx.utils.prepro import imresize


class HRNetTransform(object):

    def __init__(
            self,
            do_resize=True,
            do_normalize=False,
            size=(256, 256),
            num_of_joints=17,
            heatmap_size=(64, 64),
            sigma=2,
            mean=None,
            std=None,
            **kwargs
    ):
        self.size = size
        self.num_of_joints = num_of_joints
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.do_resize = do_resize
        self.do_normalize = do_normalize
        self.mean = mean
        self.std = std

        super(HRNetTransform, self).__init__(**kwargs)
        self.is_train = True

    def set_eval(self):
        self.is_train = False

    def set_train(self):
        self.is_train = True

    def resize(self, image, size):
        if isinstance(size, int):
            size = (size, size)
        elif isinstance(size, list):
            size = tuple(size)
        img = imresize(image, size)
        return img

    def _image_and_keypoints_process(self, image, keypoints, bbox):
        human_instance = image[bbox[1]: (bbox[1] + bbox[3]), bbox[0]: (bbox[0] + bbox[2]), :]
        left_top_of_human_instance = bbox[0:2]

        if self.do_resize:
            resize_ratio = [self.size[0] / human_instance.shape[0], self.size[1] / human_instance.shape[1]]
            image = self.resize(human_instance, (self.size[0], self.size[1]))
            # resized_image = tf.image.resize(images=human_instance, size=[self.resize_h, self.resize_w])
            for i in range(self.num_of_joints):
                if keypoints[i, 2] > 0.0:
                    keypoints[i, 0] = int((keypoints[i, 0] - left_top_of_human_instance[0]) * resize_ratio[1])
                    keypoints[i, 1] = int((keypoints[i, 1] - left_top_of_human_instance[1]) * resize_ratio[0])
        return image, keypoints

    def _get_keypoints_3d(self, keypoints):
        keypoints_3d_list = []
        keypoints_3d_exist_list = []
        for i in range(self.num_of_joints):
            keypoints_3d_list.append([keypoints[i, 0], keypoints[i, 1], 0])
            exist_value = keypoints[i, 2]
            if exist_value > 1:
                exist_value = 1
            # exist_value: (1: exist , 0: not exist)
            keypoints_3d_exist_list.append([exist_value, exist_value, 0])

        keypoints_3d = np.array(keypoints_3d_list, dtype=np.float)
        keypoints_3d_exist = np.array(keypoints_3d_exist_list, dtype=np.float)
        return keypoints_3d, keypoints_3d_exist

    def _get_one_human_instance_keypoints(self, image, annotation):
        bbox = annotation["bbox"]
        bbox = [int(i) for i in bbox]
        keypoints = annotation["keypoints"]
        keypoints = np.array(keypoints, dtype=np.float)
        keypoints_tensor = np.reshape(keypoints, newshape=(-1, 3))

        # Resize the image, and change the coordinates of the keypoints accordingly.
        image_tensor, keypoints = self._image_and_keypoints_process(image, keypoints_tensor, bbox)

        keypoints_3d, keypoints_3d_exist = self._get_keypoints_3d(keypoints)
        return image_tensor, keypoints_3d, keypoints_3d_exist

    def _generate_target(self, keypoints_3d, keypoints_3d_exist):
        target_weight = np.ones((self.num_of_joints, 1), dtype=np.float32)
        target_weight[:, 0] = keypoints_3d_exist[:, 0]

        target = np.zeros((self.num_of_joints, self.heatmap_size[0], self.heatmap_size[1]),
                          dtype=np.float32)
        temp_size = self.sigma * 3
        image_size = np.array(self.size)
        heatmap_size = np.array(self.heatmap_size)
        for joint_id in range(self.num_of_joints):
            feature_stride = image_size / heatmap_size
            mu_x = int(keypoints_3d[joint_id][0] / feature_stride[1] + 0.5)
            mu_y = int(keypoints_3d[joint_id][1] / feature_stride[0] + 0.5)
            upper_left = [int(mu_x - temp_size), int(mu_y - temp_size)]
            bottom_right = [int(mu_x + temp_size + 1), int(mu_y + temp_size + 1)]
            if upper_left[0] >= heatmap_size[1] or upper_left[1] >= heatmap_size[0] or bottom_right[0] < 0 or bottom_right[1] < 0:
                # Set the joint invisible.
                target_weight[joint_id] = 0
                continue
            size = 2 * temp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]   # shape : (size, 1)
            x0 = y0 = size // 2
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))
            g_x = max(0, -upper_left[0]), min(bottom_right[0], heatmap_size[1]) - upper_left[0]
            g_y = max(0, -upper_left[1]), min(bottom_right[1], heatmap_size[0]) - upper_left[1]
            img_x = max(0, upper_left[0]), min(bottom_right[0], heatmap_size[1])
            img_y = max(0, upper_left[1]), min(bottom_right[1], heatmap_size[0])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        target = np.transpose(target, [1, 2, 0])
        return target, target_weight

    def __call__(self, image, label, *args, **kwargs):
        image = np.array(image, dtype=np.float32)
        # max_image = np.max(image)
        # min_image = np.min(image)
        # image = (image - min_image) / (max_image - min_image)
        image = image / 255.0

        image, keypoints_3d, keypoints_3d_exist = self._get_one_human_instance_keypoints(image, label["annotations"])
        target, target_weight = self._generate_target(keypoints_3d, keypoints_3d_exist)
        return image, (target, target_weight)

