from collections import OrderedDict
from dataclasses import dataclass, fields
from typing import Optional
import tensorlayerx as tlx

if tlx.BACKEND == "tensorflow":
    import tensorflow as tf
    float_tensor = tf.Tensor
else:
    import torch
    float_tensor = torch.FloatTensor


class BaseOutput(OrderedDict):
    def __post_init__(self):
        class_fields = fields(self)

        for field in class_fields:
            v = getattr(self, field.name)
            if v is not None:
                self[field.name] = v


@dataclass
class BaseModelOutput(BaseOutput):
    output: float_tensor = None


@dataclass
class BaseTaskOutput(BaseOutput):
    logits: Optional[float_tensor] = None
    loss: Optional[float_tensor] = None


@dataclass
class BaseForImageClassificationTaskOutput(BaseTaskOutput):
    ...


@dataclass
class BaseForObjectDetectionTaskOutput(BaseTaskOutput):
    ...
