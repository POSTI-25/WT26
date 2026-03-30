import argparse
import json

from receiver import get_gpu_report, print_gpu_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Print local contributor GPU details.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON instead of human-readable output.",
    )
    args = parser.parse_args()

    report = get_gpu_report()
    if args.json:
        print(json.dumps(report, indent=2))
        return

    print_gpu_report(report)


if __name__ == "__main__":
    main()
