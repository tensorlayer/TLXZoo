from ...task.task import BaseForFaceEmbedding
from .arcface import *
from ...utils.output import BaseTaskOutput


class RetinaFaceForFaceEmbedding(BaseForFaceEmbedding):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.model = ArcFace(config.size, name="ArcFaceModel")

    def forward(self, inputs):
        return self.model(inputs)


