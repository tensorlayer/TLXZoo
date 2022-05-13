from .modeling import *
from tensorlayerx import nn
from .utils import *
from .transform import *
from scipy.optimize import linear_sum_assignment


class Detr(nn.Module):
    def __init__(self,
                 num_queries=100,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 model_dim=256,
                 backbone_bn_shape=64,
                 backbone_layer1_bn_shape=((64, 64, 256, 256), (64, 64, 256), (64, 64, 256)),
                 backbone_layer2_bn_shape=((128, 128, 512, 512), (128, 128, 512), (128, 128, 512), (128, 128, 512)),
                 backbone_layer3_bn_shape=((256, 256, 1024, 1024), (256, 256, 1024), (256, 256, 1024), (256, 256, 1024),
                                           (256, 256, 1024), (256, 256, 1024)),
                 backbone_layer4_bn_shape=((512, 512, 2048, 2048), (512, 512, 2048), (512, 512, 2048)),
                 return_intermediate_dec=True,
                 num_classes=92,
                 class_cost=1,
                 bbox_cost=5,
                 giou_cost=2,
                 num_labels=91,
                 dice_loss_coefficient=1,
                 bbox_loss_coefficient=5,
                 giou_loss_coefficient=2,
                 eos_coefficient=0.1,
                 auxiliary_loss=True,
                 name="detr", **kwargs):
        """
        :param num_queries: (:obj:`int`, `optional`, defaults to 100):
            Number of object queries, i.e. detection slots. This is the maximal number of objects
        :param num_encoder_layers: (:obj:`int`, `optional`, defaults to 6):
            Number of decoder layers.
        :param num_decoder_layers: (:obj:`int`, `optional`, defaults to 6):
            Number of encoder layers.
        :param model_dim: (:obj:`int`, `optional`, defaults to 256):
            Dimension of the layers.
        :param backbone_bn_shape: resnet bn shape
        :param backbone_layer1_bn_shape: resnet layer1 bn shape
        :param backbone_layer2_bn_shape: resnet layer2 bn shape
        :param backbone_layer3_bn_shape: resnet layer3 bn shape
        :param backbone_layer4_bn_shape: resnet layer4 bn shape
        :param return_intermediate_dec:
        :param num_classes: num of object classes + 1
        :param class_cost: (:obj:`float`, `optional`, defaults to 1):
            Relative weight of the classification error in the Hungarian matching cost.
        :param bbox_cost: (:obj:`float`, `optional`, defaults to 5):
            Relative weight of the L1 error of the bounding box coordinates in the Hungarian matching cost.
        :param giou_cost: (:obj:`float`, `optional`, defaults to 2):
            Relative weight of the generalized IoU loss of the bounding box in the Hungarian matching cost.
        :param num_labels: num of object classes
        :param dice_loss_coefficient: (:obj:`float`, `optional`, defaults to 1):
            Relative weight of the DICE/F-1 loss in the panoptic segmentation loss.
        :param bbox_loss_coefficient: (:obj:`float`, `optional`, defaults to 5):
            Relative weight of the L1 bounding box loss in the object detection loss.
        :param giou_loss_coefficient: (:obj:`float`, `optional`, defaults to 2):
            Relative weight of the generalized IoU loss in the object detection loss.
        :param eos_coefficient: (:obj:`float`, `optional`, defaults to 0.1):
            Relative classification weight of the 'no-object' class in the object detection loss.
        :param auxiliary_loss: (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether auxiliary decoding losses (loss at each decoder layer) are to be used.
        :param name:
        """
        super(Detr, self).__init__(name=name, **kwargs)
        self.num_queries = num_queries
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.model_dim = model_dim
        self.backbone_bn_shape = backbone_bn_shape
        self.backbone_layer1_bn_shape = backbone_layer1_bn_shape
        self.backbone_layer2_bn_shape = backbone_layer2_bn_shape
        self.backbone_layer3_bn_shape = backbone_layer3_bn_shape
        self.backbone_layer4_bn_shape = backbone_layer4_bn_shape
        self.return_intermediate_dec = return_intermediate_dec
        self.num_classes = num_classes
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        self.num_labels = num_labels
        self.dice_loss_coefficient = dice_loss_coefficient
        self.bbox_loss_coefficient = bbox_loss_coefficient
        self.giou_loss_coefficient = giou_loss_coefficient
        self.eos_coefficient = eos_coefficient
        self.auxiliary_loss = auxiliary_loss

        self.backbone = ResNet50Backbone(backbone_bn_shape, backbone_layer1_bn_shape, backbone_layer2_bn_shape,
                                         backbone_layer3_bn_shape, backbone_layer4_bn_shape, name=name + '/backbone')
        self.input_proj = tlx.layers.Conv2d(out_channels=model_dim, in_channels=2048,
                                            kernel_size=(1, 1), name=name + '/input_proj')
        self.query_embed = FixedEmbedding((num_queries, model_dim),
                                          name=name + '/query_embed')
        self.transformer = Transformer(
            model_dim=model_dim,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            return_intermediate_dec=return_intermediate_dec,
            name=name + '/transformer'
        )

        self.pos_encoder = PositionEmbeddingSine(
            num_pos_features=model_dim // 2, normalize=True, name=name + "/position_embedding_sine")

        self.class_embed = Linear(input_dim=model_dim, output_dim=num_classes,
                                  name='class_embed')

        self.bbox_embed_linear1 = Linear(model_dim, model_dim, name='bbox_embed_linear1')
        self.bbox_embed_linear2 = Linear(model_dim, model_dim, name='bbox_embed_linear2')
        self.bbox_embed_linear3 = Linear(model_dim, 4, name='bbox_embed_linear3')
        self.activation = tlx.ReLU()

    def loss_fn(self, m_outputs, labels):
        logits, pred_boxes = m_outputs["pred_logits"], m_outputs["pred_boxes"]
        # First: create the matcher
        matcher = DetrHungarianMatcher(
            class_cost=self.class_cost, bbox_cost=self.bbox_cost, giou_cost=self.giou_cost
        )
        # Second: create the criterion
        losses = ["labels", "boxes", "cardinality"]
        criterion = DetrLoss(
            matcher=matcher,
            num_classes=self.num_labels,
            eos_coef=self.eos_coefficient,
            losses=losses,
        )
        # Third: compute the losses, based on outputs and labels
        outputs_loss = {}
        outputs_loss["logits"] = logits
        outputs_loss["pred_boxes"] = pred_boxes
        if self.auxiliary_loss:
            outputs_loss["auxiliary_outputs"] = m_outputs["aux"]

        loss_dict = criterion(outputs_loss, labels)
        # Fourth: compute total loss, as a weighted sum of the various losses
        weight_dict = {"loss_ce": 1, "loss_bbox": self.bbox_loss_coefficient}
        weight_dict["loss_giou"] = self.giou_loss_coefficient

        if self.auxiliary_loss:
            aux_weight_dict = {}
            for i in range(self.num_decoder_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        return loss

    def downsample_masks(self, masks, x):
        # masks = tlx.cast(masks, tlx.int32)
        # masks = tlx.expand_dims(masks, -1)
        if isinstance(masks, np.ndarray):
            masks = masks.astype(np.float)
            masks = np.transpose(masks, axes=[1, 2, 0])
            masks = tlx.vision.transforms.resize(masks, tuple(x.shape[1:3]), method="nearest")
        else:
            masks = tlx.cast(masks, tlx.float32)
            masks = tlx.transpose(masks, perm=[1, 2, 0])
            # masks = tlx.resize(masks, output_size=tuple(x.shape[1:3]), method="nearest", antialias=False)
            masks = tlx.convert_to_numpy(masks)
            masks = tlx.vision.transforms.resize(masks, tuple(x.shape[1:3]), method="nearest")

        # masks = tlx.squeeze(masks, -1)
        masks = tlx.transpose(masks, perm=[2, 0, 1])
        masks = tlx.cast(masks, tlx.bool)
        return masks

    def forward(self, images, pixel_mask=None, downsample_masks=True):
        feature_maps = self.backbone(images)
        x = feature_maps[-1]

        if pixel_mask is None or tlx.get_tensor_shape(x)[0] == 1:
            pixel_mask = tlx.ones((x.shape[0], x.shape[1], x.shape[2]), tlx.bool)
        else:
            if downsample_masks:
                pixel_mask = self.downsample_masks(pixel_mask, x)

        pos_encoding = self.pos_encoder(pixel_mask)
        # feature_map = x
        projected_feature_map = self.input_proj(x)
        hs, memory = self.transformer(projected_feature_map, pixel_mask, self.query_embed(None), pos_encoding)

        transformer_output, memory, feature_maps, masks, projected_feature_map = hs, memory, feature_maps, pixel_mask, projected_feature_map

        outputs_class = self.class_embed(transformer_output)
        box_ftmps = self.activation(self.bbox_embed_linear1(transformer_output))
        box_ftmps = self.activation(self.bbox_embed_linear2(box_ftmps))
        outputs_coord = tlx.sigmoid(self.bbox_embed_linear3(box_ftmps))

        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], "memory": memory,
                  "feature_maps": feature_maps, "masks": masks, "transformer_output": transformer_output,
                  "projected_feature_map": projected_feature_map}

        output["aux"] = []
        for i in range(0, self.num_decoder_layers - 1):
            out_class = outputs_class[i]
            pred_boxes = outputs_coord[i]
            output["aux"].append({
                "logits": out_class,
                "pred_boxes": pred_boxes
            })

        return output


class DetrMaskHeadSmallConv(tlx.nn.Module):
    """
    Simple convolutional head, using group norm. Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim, name="mask_head"):
        super().__init__(name=name)

        assert (
                dim % 8 == 0
        ), "The hidden_size + number of attention heads must be divisible by 8 as the number of groups in GroupNorm is set to 8"

        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]
        self.lay1 = tlx.nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), name=name + "/lay1")
        self.gn1 = GroupNorm(dim, name=name + "/gn1")
        self.lay2 = tlx.nn.Conv2d(in_channels=dim, out_channels=inter_dims[1], kernel_size=(3, 3), name=name + "/lay2")
        self.gn2 = GroupNorm(inter_dims[1], name=name + "/gn2")
        self.lay3 = tlx.nn.Conv2d(in_channels=inter_dims[1], out_channels=inter_dims[2], kernel_size=(3, 3),
                                  name=name + "/lay3")
        self.gn3 = GroupNorm(inter_dims[2], name=name + "/gn3")
        self.lay4 = tlx.nn.Conv2d(in_channels=inter_dims[2], out_channels=inter_dims[3], kernel_size=(3, 3),
                                  name=name + "/lay4")
        self.gn4 = GroupNorm(inter_dims[3], name=name + "/gn4")
        self.lay5 = tlx.nn.Conv2d(in_channels=inter_dims[3], out_channels=inter_dims[4], kernel_size=(3, 3),
                                  name=name + "/lay5")
        self.gn5 = GroupNorm(inter_dims[4], name=name + "/gn5")
        self.out_lay = tlx.nn.Conv2d(in_channels=inter_dims[4], out_channels=1, kernel_size=(3, 3),
                                     name=name + "/out_lay")

        self.dim = dim

        self.adapter1 = tlx.nn.Conv2d(in_channels=fpn_dims[0], out_channels=inter_dims[1], kernel_size=(1, 1),
                                      name=name + "/adapter1")
        self.adapter2 = tlx.nn.Conv2d(in_channels=fpn_dims[1], out_channels=inter_dims[2], kernel_size=(1, 1),
                                      name=name + "/adapter2")
        self.adapter3 = tlx.nn.Conv2d(in_channels=fpn_dims[2], out_channels=inter_dims[3], kernel_size=(1, 1),
                                      name=name + "/adapter3")

    def forward(self, x, bbox_mask, fpns):
        # here we concatenate x, the projected feature map, of shape (batch_size, d_model, heigth/32, width/32) with
        # the bbox_mask = the attention maps of shape (batch_size, n_queries, n_heads, height/32, width/32).
        # We expand the projected feature map to match the number of heads.
        bbox_mask_shape = tlx.get_tensor_shape(bbox_mask)
        x = tlx.concat([_expand(x, bbox_mask_shape[1]),
                        tlx.reshape(bbox_mask, [-1, bbox_mask_shape[3], bbox_mask_shape[4], bbox_mask_shape[2]])], -1)

        x = self.lay1(x)
        x = self.gn1(x)
        x = tlx.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = tlx.relu(x)

        cur_fpn = self.adapter1(fpns[0])
        cur_fpn_shape = tlx.get_tensor_shape(cur_fpn)
        x_shape = tlx.get_tensor_shape(x)
        if cur_fpn_shape[0] != x_shape[0]:
            cur_fpn = _expand(cur_fpn, x_shape[0] // cur_fpn_shape[0])
        x_resize = tlx.resize(x, output_size=tuple(tlx.get_tensor_shape(cur_fpn)[1:3]),
                              method="nearest", antialias=False)
        x = cur_fpn + x_resize
        x = self.lay3(x)
        x = self.gn3(x)
        x = tlx.relu(x)

        cur_fpn = self.adapter2(fpns[1])
        cur_fpn_shape = tlx.get_tensor_shape(cur_fpn)
        x_shape = tlx.get_tensor_shape(x)
        if cur_fpn_shape[0] != x_shape[0]:
            cur_fpn = _expand(cur_fpn, x_shape[0] // cur_fpn_shape[0])
        x_resize = tlx.resize(x, output_size=tuple(tlx.get_tensor_shape(cur_fpn)[1:3]),
                              method="nearest", antialias=False)
        x = cur_fpn + x_resize
        x = self.lay4(x)
        x = self.gn4(x)
        x = tlx.relu(x)

        cur_fpn = self.adapter3(fpns[2])
        cur_fpn_shape = tlx.get_tensor_shape(cur_fpn)
        x_shape = tlx.get_tensor_shape(x)
        if cur_fpn_shape[0] != x_shape[0]:
            cur_fpn = _expand(cur_fpn, x_shape[0] // cur_fpn_shape[0])
        x_resize = tlx.resize(x, output_size=tuple(tlx.get_tensor_shape(cur_fpn)[1:3]),
                              method="nearest", antialias=False)
        x = cur_fpn + x_resize
        x = self.lay5(x)
        x = self.gn5(x)
        x = tlx.relu(x)

        x = self.out_lay(x)
        return x


class DetrMHAttentionMap(tlx.nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True, std=None, name="bbox_attention"):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.query_dim = query_dim
        # self.dropout = tlx.nn.Dropout(1 - dropout)

        self.q_linear = Linear(query_dim, hidden_dim, name=name + "/q_linear")
        self.k_linear = Linear(query_dim, hidden_dim, name=name + "/k_linear")

        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask=None):
        q = self.q_linear(q)
        k_linear_weight_unsqueeze = tlx.expand_dims(tlx.expand_dims(self.k_linear.kernel, axis=0), axis=0)
        k = tlx.ops.conv2d(k, filters=k_linear_weight_unsqueeze, strides=1, padding="SAME")
        k = tlx.add(k, self.k_linear.bias)
        # k = nn.functional.conv2d(k, k_linear_weight_unsqueeze, self.k_linear.bias)

        q_shape = tlx.get_tensor_shape(q)
        k_shape = tlx.get_tensor_shape(k)

        queries_per_head = tlx.reshape(q, [q_shape[0], q_shape[1], self.num_heads, self.hidden_dim // self.num_heads])
        keys_per_head = tlx.reshape(k, [k_shape[0], self.num_heads, self.hidden_dim // self.num_heads,
                                        k_shape[1], k_shape[2]])
        queries_per_head = queries_per_head * self.normalize_fact

        weights = einsum("bqnc,bnchw->bqnhw", queries_per_head, keys_per_head)

        if mask is not None:
            mask = tlx.expand_dims(tlx.expand_dims(mask, axis=1), axis=1)
            weights = tlx.where(mask, weights, float("-inf"))
            # weights.masked_fill_(mask, float("-inf"))
        shape = tlx.get_tensor_shape(weights)
        weights = tlx.reshape(weights, [shape[0], shape[1], -1])
        weights = tlx.softmax(weights, axis=-1)
        weights = tlx.reshape(weights, shape)
        # weights = self.dropout(weights)
        return weights


def _expand(tensor, length):
    tensor = tlx.expand_dims(tensor, 1)
    tensor = tlx.tile(tensor, [1, int(length), 1, 1, 1])
    shape = tlx.get_tensor_shape(tensor)
    tensor = tlx.reshape(tensor, [-1, shape[2], shape[3], shape[4]])
    return tensor


class DetrHungarianMatcher(tlx.nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general, there are more
    predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).
    """

    def __init__(self, class_cost: float = 1, bbox_cost: float = 1, giou_cost: float = 1):
        """
        Creates the matcher.

        Params:
            class_cost: This is the relative weight of the classification error in the matching cost
            bbox_cost: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            giou_cost: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()

        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        assert class_cost != 0 or bbox_cost != 0 or giou_cost != 0, "All costs of the Matcher can't be 0"

    def forward(self, outputs, targets):
        """
        Performs the matching.

        Params:
            outputs: This is a dict that contains at least these entries:
                 "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                 objects in the target) containing the class labels "boxes": Tensor of dim [num_target_boxes, 4]
                 containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:

                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries, num_classes = tlx.get_tensor_shape(outputs["logits"])

        # We flatten to compute the cost matrices in a batch

        out_prob = tlx.reshape(outputs["logits"], [bs * num_queries, num_classes])
        out_prob = tlx.softmax(out_prob, axis=-1)  # [batch_size * num_queries, num_classes]
        out_bbox = tlx.reshape(outputs["pred_boxes"], [bs * num_queries, 4])  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = tlx.concat([v["class_labels"] for v in targets], axis=0)
        tgt_bbox = tlx.concat([v["boxes"] for v in targets], axis=0)

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        class_cost = -tlx.gather(out_prob, tgt_ids, axis=1)

        # Compute the L1 cost between boxes
        bbox_cost = cdist(out_bbox, tgt_bbox)

        # Compute the giou cost between boxes
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(tgt_bbox))

        # Final cost matrix
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = tlx.reshape(cost_matrix, [bs, num_queries, -1])
        # cost_matrix = cost_matrix.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(tlx.split(cost_matrix, sizes, -1))]
        return [(tlx.convert_to_tensor(i, dtype=tlx.int64), tlx.convert_to_tensor(j, dtype=tlx.int64)) for i, j in
                indices]


class DetrLoss(tlx.nn.Module):
    """
    This class computes the losses for DetrForObjectDetection/DetrForSegmentation. The process happens in two steps: 1)
    we compute hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each pair
    of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, num_classes, eos_coef, losses):
        """
        Create the criterion.

        A note on the num_classes parameter (copied from original repo in detr.py): "the naming of the `num_classes`
        parameter of the criterion is somewhat misleading. it indeed corresponds to `max_obj_id + 1`, where max_obj_id
        is the maximum id for a class in your dataset. For example, COCO has a max_obj_id of 90, so we pass
        `num_classes` to be 91. As another example, for a dataset that has a single class with id 1, you should pass
        `num_classes` to be 2 (max_obj_id + 1). For more details on this, check the following discussion
        https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223"

        Parameters:
            matcher: module able to compute a matching between targets and proposals.
            num_classes: number of object categories, omitting the special no-object category.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = np.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.empty_weight = tlx.convert_to_tensor(empty_weight, dtype=tlx.float32)

    # removed logging parameter, which was part of the original implementation
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (NLL) targets dicts must contain the key "class_labels" containing a tensor of dim
        [nb_target_boxes]
        """
        assert "logits" in outputs, "No logits were found in the outputs"
        src_logits = outputs["logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = tlx.concat([tlx.gather(t["class_labels"], J, axis=0) for t, (_, J) in zip(targets, indices)],
                                      axis=0)
        target_classes = np.ones(src_logits.shape[:2]) * self.num_classes
        target_classes[tlx.convert_to_numpy(idx[0]), tlx.convert_to_numpy(idx[1])] = target_classes_o
        target_classes = tlx.convert_to_tensor(target_classes, dtype=tlx.int64)

        weight = tlx.gather(self.empty_weight, target_classes)
        loss = tlx.losses.softmax_cross_entropy_with_logits(tlx.reshape(src_logits, [-1, (self.num_classes + 1)]),
                                                            tlx.reshape(target_classes, [-1]), reduction="none")
        loss *= weight

        loss_ce = tlx.reduce_mean(loss)

        losses = {"loss_ce": loss_ce}

        return losses

    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.

        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        logits = outputs["logits"]
        tgt_lengths = tlx.convert_to_tensor([len(v["class_labels"]) for v in targets])
        # Count the number of predictions that are NOT "no-object" (which is the last class)

        arg = tlx.argmax(logits, axis=-1)
        arg = arg != self.num_classes
        arg = tlx.cast(arg, tlx.int32)
        card_pred = tlx.reduce_sum(arg)

        card_pred = tlx.cast(card_pred, tlx.float32)
        tgt_lengths = tlx.cast(tgt_lengths, tlx.float32)
        card_err = tlx.abs(card_pred - tgt_lengths)
        card_err = tlx.reduce_sum(card_err)
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs, "No predicted boxes found in outputs"
        idx = self._get_src_permutation_idx(indices)
        idx = tlx.stack(idx, axis=0)
        idx = tlx.transpose(idx)
        src_boxes = tlx.gather_nd(outputs["pred_boxes"], idx)
        target_boxes = tlx.concat([tlx.gather(t["boxes"], i, axis=0) for t, (_, i) in zip(targets, indices)], axis=0)

        l1_loss = tlx.abs(src_boxes - target_boxes)

        losses = {}
        losses["loss_bbox"] = tlx.reduce_sum(l1_loss) / num_boxes

        loss_giou = generalized_box_iou(center_to_corners_format(src_boxes), center_to_corners_format(target_boxes))
        shape = tlx.get_tensor_shape(loss_giou)[0]
        shape = tlx.arange(shape)
        shape = tlx.stack([shape, shape])
        shape = tlx.transpose(shape)
        loss_giou = tlx.gather_nd(loss_giou, shape)

        loss_giou = 1 - loss_giou
        losses["loss_giou"] = tlx.reduce_sum(loss_giou) / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.

        Targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w].
        """
        assert "pred_masks" in outputs, "No predicted masks found in outputs"

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_idx = tlx.stack(src_idx, axis=0)
        src_idx = tlx.transpose(src_idx)
        src_masks = tlx.gather_nd(src_masks, src_idx)
        masks = [t["masks"] for t in targets]

        target_masks, valid = nested_tensor_from_tensor_list(masks)
        tgt_idx = tlx.stack(tgt_idx, axis=0)
        tgt_idx = tlx.transpose(tgt_idx)
        target_masks = tlx.gather_nd(target_masks, tgt_idx)

        # upsample predictions to the target size
        target_masks_shape = tlx.get_tensor_shape(target_masks)
        src_masks = tlx.transpose(src_masks, perm=[1, 2, 0])
        src_masks = tlx.resize(src_masks, output_size=tuple(target_masks_shape[-2:]),
                               method="bilinear", antialias=False)
        src_masks = tlx.transpose(src_masks, perm=[2, 0, 1])
        src_masks_shape = tlx.get_tensor_shape(src_masks)
        src_masks = tlx.reshape(src_masks, [src_masks_shape[0], -1])

        target_masks = tlx.reshape(target_masks, tlx.get_tensor_shape(src_masks))
        target_masks = tlx.cast(target_masks, dtype=tlx.float32)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = tlx.concat([tlx.ones_like(src) * i for i, (src, _) in enumerate(indices)], axis=0)
        src_idx = tlx.concat([src for (src, _) in indices], axis=0)
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = tlx.concat([tlx.ones_like(tgt) * i for i, (_, tgt) in enumerate(indices)], axis=0)
        tgt_idx = tlx.concat([tgt for (_, tgt) in indices], axis=0)
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"Loss {loss} not supported"
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        """
        This performs the loss computation.

        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = tlx.convert_to_tensor([num_boxes], dtype=tlx.float32)
        # (Niels): comment out function below, distributed training to be added
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        # (Niels) in original implementation, num_boxes is divided by get_world_size()
        num_boxes = tlx.where(num_boxes >= 1, num_boxes, 1)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                indices = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/. The boxes should be in [x0, y0, x1, y1] (corner) format.

    Returns:
        a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    # assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = tlx.minimum(boxes1[:, None, :2], boxes2[:, :2])
    rb = tlx.maximum(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = rb - lt
    wh = tlx.where(wh >= 0, wh, 0)
    # wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = tlx.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = tlx.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = rb - lt
    wh = tlx.where(wh >= 0, wh, 0)
    # wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def _upcast(t):
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (tlx.float32, tlx.float64) else t.float()
    else:
        return t if t.dtype in (tlx.int32, tlx.int64) else t.int()


def nested_tensor_from_tensor_list(tensor_list):
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])

        tensors = []
        masks = []
        for img in tensor_list:
            img = tlx.convert_to_tensor(img)
            img_shape = tlx.get_tensor_shape(img)
            new_img = tlx.pad(img, paddings=[[0, max_size[0] - img_shape[0]], [0, max_size[1] - img_shape[1]],
                                             [0, max_size[2] - img_shape[2]]])
            tensors.append(new_img)
            mask = tlx.zeros((img_shape[1], img_shape[2]))
            new_mask = tlx.pad(mask, paddings=[[0, max_size[1] - img_shape[1]],
                                               [0, max_size[2] - img_shape[2]]], constant_values=1)
            new_mask = tlx.cast(new_mask, tlx.bool)
            masks.append(new_mask)
        tensor = tlx.stack(tensors, axis=0)
        mask = tlx.stack(masks, axis=0)
    else:
        raise ValueError("Only 3-dimensional tensors are supported")
    return tensor, mask


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs (0 for the negative class and 1 for the positive
                 class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.

    Returns:
        Loss tensor
    """
    prob = tlx.sigmoid(inputs)
    ce_loss = tlx.losses.binary_cross_entropy(prob, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return tlx.reduce_sum(tlx.reduce_mean(loss, axis=1)) / num_boxes


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs (0 for the negative class and 1 for the positive
                 class).
    """
    inputs = tlx.sigmoid(inputs)
    numerator = tlx.reduce_sum(2 * (inputs * targets), axis=1)
    denominator = tlx.reduce_sum(inputs, axis=-1) + tlx.reduce_sum(targets, axis=-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return tlx.reduce_sum(loss) / num_boxes
