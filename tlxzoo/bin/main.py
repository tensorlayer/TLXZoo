import argparse
import sys


def run(args):
    ...


def _get_augment_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    return parser


def main():
    parser = _get_augment_parser()
    args = sys.argv[1:]
    args = parser.parse_args(args)
    run(args)


if __name__ == '__main__':
    main()

