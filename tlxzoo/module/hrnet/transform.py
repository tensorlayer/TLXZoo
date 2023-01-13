import numpy as np
import cv2


class Gather(object):
    def __call__(self, data):
        image = np.array(data['image'])
        bbox = data['annotations']['bbox']
        bbox = [int(i) for i in bbox]
        keypoints = data['annotations']['keypoints']
        keypoints = np.array(keypoints, dtype=np.int)
        keypoints = np.reshape(keypoints, newshape=(-1, 3))
        return {
            'image': image,
            'bbox': bbox,
            'keypoints': keypoints
        }


class Crop(object):
    def __call__(self, data):
        image, bbox, keypoints = data['image'], data['bbox'], data['keypoints']

        image = image[bbox[1]: (bbox[1] + bbox[3]), bbox[0]: (bbox[0] + bbox[2]), :]
        mask = keypoints[:, 2] > 0
        keypoints[:, 0][mask] -= bbox[0]
        keypoints[:, 1][mask] -= bbox[1]

        return {
            'image': image,
            'keypoints': keypoints
        }
        
    
class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, keypoints = data['image'], data['keypoints'].astype(np.float32)
        origin_shape = image.shape[:2]
        
        image = cv2.resize(image, self.size)
        keypoints[:, 0] *= self.size[1] / origin_shape[1]
        keypoints[:, 1] *= self.size[0] / origin_shape[0]
        
        return {
            'image': image,
            'keypoints': keypoints.astype(np.int)
        }


class Normalize(object):
    def __init__(self, mean=(0.0, 0.0, 0.0), std=(255.0, 255.0, 255.0)):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
    
    def __call__(self, data):
        data['image'] = (data['image'].astype(np.float32) - self.mean) / self.std
        return data


class GenerateTarget(object):

    def __init__(
            self,
            size=(256, 256),
            num_of_joints=17,
            heatmap_size=(64, 64),
            sigma=2,
            **kwargs
    ):
        self.size = size
        self.num_of_joints = num_of_joints
        self.heatmap_size = heatmap_size
        self.sigma = sigma


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

    def __call__(self, data, *args, **kwargs):
        image, keypoints = data['image'], data['keypoints']
        keypoints_3d, keypoints_3d_exist = self._get_keypoints_3d(keypoints)
        target, target_weight = self._generate_target(keypoints_3d, keypoints_3d_exist)
        return {
            'image': image,
            'target': target,
            'target_weight': target_weight
        }

