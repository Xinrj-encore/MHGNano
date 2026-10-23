import argparse

from maml_statistics import add_statistical_args


def build_parser():
    parser = argparse.ArgumentParser(description="Train and evaluate MHGNano across species")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--kingdom", choices=("eukaryotes", "prokaryotes"))
    parser.add_argument(
        "--species-dir",
        action="append",
        default=[],
        metavar="SPECIES=PATH",
        help="Use an exact species data directory; repeat for mixed-domain runs.",
    )
    parser.add_argument("--train-species", nargs="+", required=True)
    parser.add_argument("--validation-species", required=True)
    parser.add_argument("--test-species", nargs="+", required=True)
    parser.add_argument("--model-size", default="8M", choices=("8M", "35M", "150M", "650M"))
    parser.add_argument("--in-channels", type=int, default=320)
    parser.add_argument("--hid-channels", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=128)
    parser.add_argument("--inner-lr", type=float, default=1e-2)
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--meta-lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--meta-batch-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="./output")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument(
        "--neg-sampling",
        default="original",
        choices=("original", "existing_random", "degree_matched"),
    )
    parser.add_argument("--neg-ratio", type=float, default=1.0)
    parser.add_argument("--use-val-threshold", action="store_true")
    parser.add_argument(
        "--loss-type",
        default="bce",
        choices=("bce", "weighted_bce", "focal", "weighted_focal", "bce_pairwise", "pairwise_auc"),
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--pairwise-lambda", type=float, default=0.2)
    parser.add_argument("--pairwise-margin", type=float, default=0.0)
    parser.add_argument("--pairwise-max-pairs", type=int, default=4096)
    add_statistical_args(parser)
    return parser


def main():
    args = build_parser().parse_args()
    from .experiment import run_experiment

    run_experiment(args)
