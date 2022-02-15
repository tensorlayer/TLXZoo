import argparse
import sys
from tlxzoo.config import BaseRunnerConfig
from tlxzoo.task import BaseTask
from tlxzoo.dataset import DataLoaders
from tlxzoo.feature import BaseFeature
from tlxzoo.trainer import Trainer


def run(args):
    config_path = args.config
    runner_config = BaseRunnerConfig.from_pretrained(config_path)

    load_weight = args.load_weight
    task = BaseTask.from_pretrained(config_path, config=runner_config.task_config, load_weight=load_weight)

    data_loaders = DataLoaders(runner_config.data_config)

    feature = BaseFeature.from_config(runner_config.feature_config)

    trainer = Trainer(task=task, data_loader=data_loaders, config=runner_config.trainer_config)
    trainer.register_feature_transform_hook(feature)

    print_freq = args.print_freq
    print_train_batch = args.print_train_batch
    trainer.train(n_epoch=runner_config.trainer_config.epochs, print_freq=print_freq, print_train_batch=print_train_batch)

    if args.save:
        runner_config.save_pretrained(args.save_dir)
        trainer.save(args.save_dir)


def _get_augment_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", required=True, help="Path of the main config file."
    )

    parser.add_argument(
        "--save_dir", help="Path for save config or weight."
    )

    parser.add_argument(
        "--save",
        default=False,
        action="store_true",
        help="save model weight and weight.",
    )

    parser.add_argument(
        "--load_weight",
        default=False,
        action="store_true",
        help="load model weight.",
    )

    parser.add_argument(
        "--print_freq",
        type=int,
        default=2,
        help="print log for n epochs.",
    )

    parser.add_argument(
        "--print_train_batch",
        default=False,
        action="store_true",
        help="whether print log every batch.",
    )

    return parser


def main():
    parser = _get_augment_parser()
    args = sys.argv[1:]
    args = parser.parse_args(args)
    run(args)


if __name__ == '__main__':
    main()

