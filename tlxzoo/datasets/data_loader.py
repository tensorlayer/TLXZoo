from .dataset import IterableDataset
import tensorlayerx as tlx
from ..utils.registry import Registers
import copy


class DataLoaders(object):
    def __init__(self, data_name, train_limit=None, collate_fn=None, transform_hook=None, transform_hook_index=None,
                 num_workers=0, per_device_train_batch_size=2, per_device_eval_batch_size=2, **kwargs):
        self.dataset_dict = Registers.datasets[data_name].load(train_limit=train_limit, **kwargs)

        if "train" in self.dataset_dict:
            train_data = self.dataset_dict["train"]
            if transform_hook:
                transform_hook = copy.deepcopy(transform_hook)
                transform_hook.set_train()
                train_data.register_transform_hook(transform_hook, index=transform_hook_index)
            self.train = self.dataset_dataloader(train_data, dataset_type="train", collate_fn=collate_fn,
                                                 num_workers=num_workers,
                                                 per_device_train_batch_size=per_device_train_batch_size,
                                                 per_device_eval_batch_size=per_device_eval_batch_size)
        else:
            self.train = None

        if "eval" in self.dataset_dict:
            eval_data = self.dataset_dict["eval"]

            if transform_hook:
                transform_hook = copy.deepcopy(transform_hook)
                transform_hook.set_eval()
                eval_data.register_transform_hook(transform_hook, index=transform_hook_index)
            self.eval = self.dataset_dataloader(eval_data, collate_fn=collate_fn,
                                                dataset_type="eval",
                                                per_device_train_batch_size=per_device_train_batch_size,
                                                per_device_eval_batch_size=per_device_eval_batch_size
                                                )
        else:
            self.eval = None

        if "test" in self.dataset_dict:
            test_data = self.dataset_dict["test"]
            if transform_hook:
                transform_hook = copy.deepcopy(transform_hook)
                transform_hook.set_eval()
                test_data.register_transform_hook(transform_hook, index=transform_hook_index)
            self.test = self.dataset_dataloader(test_data, collate_fn=collate_fn,
                                                dataset_type="test",
                                                per_device_train_batch_size=per_device_train_batch_size,
                                                per_device_eval_batch_size=per_device_eval_batch_size
                                                )
        else:
            self.test = None

    def register_transform_hook(self, transform_hook, index=None):

        if "train" in self.dataset_dict:
            transform_hook = copy.deepcopy(transform_hook)
            transform_hook.set_train()
            self.dataset_dict["train"].register_transform_hook(transform_hook, index=index)

        if "eval" in self.dataset_dict:
            transform_hook = copy.deepcopy(transform_hook)
            transform_hook.set_eval()
            self.dataset_dict["eval"].register_transform_hook(transform_hook, index=index)

        if "test" in self.dataset_dict:
            transform_hook = copy.deepcopy(transform_hook)
            transform_hook.set_eval()
            self.dataset_dict["test"].register_transform_hook(transform_hook, index=index)

    def dataset_dataloader(self, dataset, dataset_type="train", num_workers=0, collate_fn=None,
                           per_device_train_batch_size=2, per_device_eval_batch_size=2):

        if dataset_type == "train":
            if isinstance(dataset, IterableDataset):
                shuffle = False
            else:
                shuffle = True
            if num_workers == 0:
                train_loader = tlx.dataflow.DataLoader(dataset,
                                                       batch_size=per_device_train_batch_size,
                                                       collate_fn=collate_fn,
                                                       shuffle=shuffle)
            else:
                train_loader = tlx.dataflow.DataLoader(dataset,
                                                       batch_size=per_device_train_batch_size,
                                                       prefetch_factor=per_device_train_batch_size,
                                                       num_workers=num_workers,
                                                       collate_fn=collate_fn,
                                                       shuffle=shuffle)
            return train_loader
        else:
            return tlx.dataflow.DataLoader(dataset, collate_fn=collate_fn,
                                           batch_size=per_device_eval_batch_size, shuffle=False)