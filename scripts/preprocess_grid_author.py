#!/usr/bin/env python3
"""Run the local GRID author-code CLIP preprocessor on converted HSG-RTP data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path, default=Path("/home/swzz/disk2T/grid/ridsg"))
    parser.add_argument("--config", type=Path, default=Path("hparams_hsg_rtp.cfg"))
    parser.add_argument("--raw-data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    author_root = args.author_root.resolve()
    config_path = args.config if args.config.is_absolute() else author_root / args.config
    sys.path.insert(0, str(author_root))

    from arguments import Config
    from dataloader.data_preprocessor import data_preprocessor

    config = Config(str(config_path))
    config.output_preprocess_path = str(args.output.resolve())
    config.dataset_size = args.limit
    config.batch_size = args.batch_size
    device = int(args.device) if args.device.isdigit() else args.device
    runtime_args = SimpleNamespace(gpu_devices=[device])
    processor = data_preprocessor(arg=runtime_args, config=config)
    processor.preprocess_from_file_by_batch(
        data_path=str(args.raw_data.resolve()),
        resume_count_flag=args.resume,
    )


if __name__ == "__main__":
    main()
