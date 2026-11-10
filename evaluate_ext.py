import argparse
import functools

import evaluate
import models_ext
import utils

evaluate.build_model = models_ext.build_model


def main(argv=None, batch_size=None):
    if batch_size:
        evaluate.predict = functools.partial(utils.predict, batch_size=batch_size)
    return evaluate.main(argv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--batch-size", type=int, default=None)
    args, rest = parser.parse_known_args()
    main(rest, args.batch_size)
