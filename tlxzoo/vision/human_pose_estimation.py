from tlxzoo.module import *
import tensorlayerx as tlx
import cv2


class HumanPoseEstimation(tlx.nn.Module):
    def __init__(self, backbone, **kwargs):
        super(HumanPoseEstimation, self).__init__()
        if backbone == "hrnet":
            self.backbone = PoseHighResolutionNet(**kwargs)
        else:
            raise ValueError(f"tlxzoo don`t support {backbone}")

    def loss_fn(self, output, name="", **kwargs):
        if hasattr(self.backbone, "loss_fn"):
            return self.backbone.loss_fn(output, **kwargs)
        else:
            raise ValueError("loss fn isn't defined.")

    def forward(self, inputs):
        return self.backbone(inputs)


def get_final_preds(batch_heatmaps):
    preds, maxval = get_max_preds(batch_heatmaps)
    num_of_joints = preds.shape[-1]
    batch_size = preds.shape[0]
    batch_x = []
    batch_y = []
    for b in range(batch_size):
        single_image_x = []
        single_image_y = []
        for j in range(num_of_joints):
            point_x = int(preds[b, 0, j])
            point_y = int(preds[b, 1, j])
            single_image_x.append(point_x)
            single_image_y.append(point_y)
        batch_x.append(single_image_x)
        batch_y.append(single_image_y)
    return batch_x, batch_y


def get_dye_vat_bgr():
    DYE_VAT = {"Pink": (255, 192, 203), "MediumVioletRed": (199, 21, 133), "Magenta": (255, 0, 255),
               "Purple": (128, 0, 128), "Blue": (0, 0, 255), "LightSkyBlue": (135, 206, 250),
               "Cyan": (0, 255, 255), "LightGreen": (144, 238, 144), "Green": (0, 128, 0),
               "Yellow": (255, 255, 0), "Gold": (255, 215, 0), "Orange": (255, 165, 0),
               "Red": (255, 0, 0), "LightCoral": (240, 128, 128), "DarkGray": (169, 169, 169)}
    bgr_color = {}
    for k, v in DYE_VAT.items():
        r, g, b = v[0], v[1], v[2]
        bgr_color[k] = (b, g, r)
    return bgr_color


def color_pool():
    bgr_color_dict = get_dye_vat_bgr()
    bgr_color_pool = []
    for k, v in bgr_color_dict.items():
        bgr_color_pool.append(v)
    return bgr_color_pool


def draw_on_image(image, x, y, rescale):
    keypoints_coords = []
    for j in range(len(x)):
        x_coord, y_coord = rescale(x=x[j], y=y[j])
        keypoints_coords.append([x_coord, y_coord])
        cv2.circle(img=image, center=(x_coord, y_coord), radius=8, color=get_dye_vat_bgr()["Red"], thickness=2)
    # draw lines
    color_list = color_pool()
    SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    for i in range(len(SKELETON)):
        index_1 = SKELETON[i][0] - 1
        index_2 = SKELETON[i][1] - 1
        x1, y1 = rescale(x=x[index_1], y=y[index_1])
        x2, y2 = rescale(x=x[index_2], y=y[index_2])
        cv2.line(img=image, pt1=(x1, y1), pt2=(x2, y2), color=color_list[i % len(color_list)], thickness=5, lineType=cv2.LINE_AA)
    return image


def inference(image_tensor, model, image_dir, original_image_size):
    model.set_eval()
    pred_heatmap = model(image_tensor)
    keypoints_rescale = KeypointsRescaleToOriginal(input_image_height=256,
                                                   input_image_width=256,
                                                   heatmap_h=pred_heatmap.shape[1],
                                                   heatmap_w=pred_heatmap.shape[2],
                                                   original_image_size=original_image_size)
    batch_x_list, batch_y_list = get_final_preds(batch_heatmaps=pred_heatmap)
    keypoints_x = batch_x_list[0]
    keypoints_y = batch_y_list[0]
    image = draw_on_image(image=cv2.imread(image_dir), x=keypoints_x, y=keypoints_y, rescale=keypoints_rescale)

    # cv2.namedWindow("Pose Estimation", flags=cv2.WINDOW_NORMAL)
    # cv2.imshow("Pose Estimation", image)
    # cv2.waitKey(0)
    cv2.imwrite("image.png", image)


class KeypointsRescaleToOriginal(object):
    def __init__(self, input_image_height, input_image_width, heatmap_h, heatmap_w, original_image_size):
        self.scale_ratio = [input_image_height / heatmap_h, input_image_width / heatmap_w]
        self.original_scale_ratio = [original_image_size[0] / input_image_height, original_image_size[1] / input_image_width]

    def __scale_to_input_size(self, x, y):
        return x * self.scale_ratio[1], y * self.scale_ratio[0]

    def __call__(self, x, y):
        temp_x, temp_y = self.__scale_to_input_size(x=x, y=y)
        return int(temp_x * self.original_scale_ratio[1]), int(temp_y * self.original_scale_ratio[0])




