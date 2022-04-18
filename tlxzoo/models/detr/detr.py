from ..model import BaseModule
from .modeling import *


class DetrModel(BaseModule):
    def __init__(self, config, name="detr",
                 **kwargs):
        super(DetrModel, self).__init__(config, name=name, **kwargs)
        self.num_queries = config.num_queries

        self.backbone = ResNet50Backbone(config, name=name+'/backbone')
        self.input_proj = tlx.layers.Conv2d(out_channels=config.model_dim, in_channels=2048,
                                            kernel_size=(1, 1), name=name+'/input_proj')
        self.query_embed = FixedEmbedding((config.num_queries, config.model_dim),
                                          name=name+'/query_embed')
        self.transformer = Transformer(
            model_dim=config.model_dim,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            return_intermediate_dec=config.return_intermediate_dec,
            name=name+'/transformer'
        )

        self.pos_encoder = PositionEmbeddingSine(
            num_pos_features=config.model_dim // 2, normalize=True, name=name+"/position_embedding_sine")

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

    def forward(self, images, masks=None, downsample_masks=True):
        feature_maps = self.backbone(images)
        x = feature_maps[-1]

        if masks is None or tlx.get_tensor_shape(x)[0] == 1:
            masks = tlx.ones((x.shape[0], x.shape[1], x.shape[2]), tlx.bool)
        else:
            if downsample_masks:
                masks = self.downsample_masks(masks, x)

        pos_encoding = self.pos_encoder(masks)
        # feature_map = x
        projected_feature_map = self.input_proj(x)
        hs, memory = self.transformer(projected_feature_map, masks, self.query_embed(None), pos_encoding)

        return hs, memory, feature_maps, masks, projected_feature_map
