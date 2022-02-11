import argparse
import sys
from tlxzoo.config import BaseAppConfig
from tlxzoo.task import BaseTask
from tlxzoo.dataset import DataLoaders


def run(args):
    config_path = args.config
    app_config = BaseAppConfig.from_pretrained(config_path)

    load_weight = args.load_weight
    task = BaseTask.from_pretrained(config_path, config=app_config.task_config)

    data_loaders = DataLoaders(app_config.data_config)




def _get_augment_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", required=True, help="Path of the main config file."
    )

    parser.add_argument(
        "--load_weight",
        default=False,
        action="store_true",
        help="load model weight.",
    )

    return parser


def main():
    parser = _get_augment_parser()
    args = sys.argv[1:]
    args = parser.parse_args(args)
    run(args)


if __name__ == '__main__':
    main()

