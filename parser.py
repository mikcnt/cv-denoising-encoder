import argparse


def main_parser():
    # Parse arguments and prepare program
    parser = argparse.ArgumentParser(description="Arguments parser")
    parser.add_argument(
        "--model_checkpoint",
        default="",
        type=str,
        help="path to .pth file checkpoint of the model (default: none)",
    )
    parser.add_argument(
        "--resume_last",
        dest="resume_last",
        action="store_true",
        help="use this flag to resume the last checkpoint of the model",
    )
    parser.add_argument(
        "--batch_size", default=8, type=int, help="batch size (default: 8)"
    )
    parser.add_argument(
        "--epochs", default=500, type=int, help="number of epochs (default: 500)"
    )
    parser.add_argument(
        "--learning_rate",
        default=0.0003,
        type=float,
        help="learning rate (default 0.1)",
    )
    parser.add_argument(
        "--data_path",
        default="data",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--use_drive",
        dest="use_drive",
        action="store_true",
        help="use this flag to save checkpoint on drive",
    )
    return parser