from ...config.config import BaseModelConfig, BaseTaskConfig
from ...utils import MODEL_WEIGHT_NAME, TASK_WEIGHT_NAME
from ...utils.registry import Registers


class StageParams(object):
    def __init__(self, channels, modules, block, num_blocks, fusion_method):
        self.channels = channels
        self.modules = modules
        self.block = block
        self.num_blocks = num_blocks
        self.fusion_method = fusion_method
        self.expansion = self.__get_expansion()

    def __get_expansion(self):
        if self.block == "BASIC":
            return 1
        elif self.block == "BOTTLENECK":
            return 4
        else:
            raise ValueError("Invalid block name.")

    def get_stage_channels(self):
        num_channels = [num_channel * self.expansion for num_channel in self.channels]
        return num_channels

    def get_branch_num(self):
        return len(self.channels)

    def get_modules(self):
        return self.modules

    def get_block(self):
        return self.block

    def get_num_blocks(self):
        return self.num_blocks

    def get_fusion_method(self):
        return self.fusion_method


@Registers.model_configs.register
class HRNetModelConfig(BaseModelConfig):
    model_type = "hrnet"

    def __init__(
            self,
            conv3_kernel=3,
            weights_path=MODEL_WEIGHT_NAME,
            **kwargs
    ):
        self.conv3_kernel = conv3_kernel
        self.num_of_joints = 17
        self.SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                         [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

        self.stage_2 = StageParams(channels=[32, 64], modules=1, block="BASIC", num_blocks=[4, 4], fusion_method="sum")
        self.stage_3 = StageParams(channels=[32, 64, 128], modules=4, block="BASIC", num_blocks=[4, 4, 4],
                                   fusion_method="sum")
        self.stage_4 = StageParams(channels=[32, 64, 128, 256], modules=3, block="BASIC", num_blocks=[4, 4, 4, 4],
                                   fusion_method="sum")

        super().__init__(
            weights_path=weights_path,
            **kwargs,
        )

    def get_stage(self, stage_name):
        if stage_name == "s2":
            channels = self.stage_2.get_stage_channels()
            num_branches = self.stage_2.get_branch_num()
            num_modules = self.stage_2.get_modules()
            block = self.stage_2.get_block()
            num_blocks = self.stage_2.get_num_blocks()
            fusion_method = self.stage_2.get_fusion_method()
        elif stage_name == "s3":
            channels = self.stage_3.get_stage_channels()
            num_branches = self.stage_3.get_branch_num()
            num_modules = self.stage_3.get_modules()
            block = self.stage_3.get_block()
            num_blocks = self.stage_3.get_num_blocks()
            fusion_method = self.stage_3.get_fusion_method()
        elif stage_name == "s4":
            channels = self.stage_4.get_stage_channels()
            num_branches = self.stage_4.get_branch_num()
            num_modules = self.stage_4.get_modules()
            block = self.stage_4.get_block()
            num_blocks = self.stage_4.get_num_blocks()
            fusion_method = self.stage_4.get_fusion_method()
        else:
            raise ValueError("Invalid stage name.")
        return [channels, num_branches, num_modules, block, num_blocks, fusion_method]


@Registers.task_configs.register
class TrOCRForHumanPoseEstimationTaskConfig(BaseTaskConfig):
    task_type = "hrnet_for_human_pose_estimation"
    model_config_type = HRNetModelConfig

    def __init__(self,
                 model_config: model_config_type = None,
                 weights_path=TASK_WEIGHT_NAME,
                 **kwargs):
        if model_config is None:
            model_config = self.model_config_type()

        if weights_path is None:
            self.weights_path = model_config.weights_path
        else:
            self.weights_path = weights_path
        super(TrOCRForHumanPoseEstimationTaskConfig, self).__init__(model_config, **kwargs)