import argparse
import os

import torch


def parse_args():
    parser = argparse.ArgumentParser("Convert Swin Transformer to Detectron2")

    parser.add_argument("source_model", default="", type=str,
                        help="Source model")
    parser.add_argument("output_model", default="", type=str,
                        help="Output model")
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.splitext(args.source_model)[-1] != ".pth":
        raise ValueError("You should save weights as pth file")

    source_weights = torch.load(
        args.source_model, map_location=torch.device('cpu'))["model"]
    converted_weights = {}
    keys = list(source_weights.keys())

    #prefix = 'backbone.bottom_up.'
    prefix = 'student.raw_backbone.'
    for key in keys:
        converted_weights[prefix+key] = source_weights[key]

    torch.save(converted_weights, args.output_model)


if __name__ == "__main__":
    main()
