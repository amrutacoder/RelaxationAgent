"""Script to download and prepare GoEmotions dataset."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.milestone_b.goemotions_loader import GoEmotionsLoader


def main():
    parser = argparse.ArgumentParser(description="Download GoEmotions dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/goemotions",
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="Dataset splits to download"
    )
    parser.add_argument(
        "--simplified",
        action="store_true",
        default=True,
        help="Use simplified version (54k vs 211k rows)"
    )
    parser.add_argument(
        "--no-simplified",
        dest="simplified",
        action="store_false",
        help="Use full version (211k rows)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples per split (for testing)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GoEmotions Dataset Downloader")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Splits: {args.splits}")
    print(f"Simplified: {args.simplified}")
    print(f"Max samples per split: {args.max_samples or 'All'}")
    print()
    
    try:
        loader = GoEmotionsLoader()
        loader.download_and_save(
            output_dir=args.output_dir,
            splits=args.splits,
            simplified=args.simplified,
            max_samples_per_split=args.max_samples
        )
        
        print("\n" + "=" * 60)
        print("Download complete!")
        print("=" * 60)
        print(f"\nNext steps:")
        print(f"1. Review downloaded files in: {args.output_dir}")
        print(f"2. Train text emotion classifier:")
        print(f"   python -m src.milestone_c.train_text --data_dir {args.output_dir}")
        
    except ImportError as e:
        print(f"\nError: {e}")
        print("\nPlease install required dependencies:")
        print("  pip install datasets")
        sys.exit(1)
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
