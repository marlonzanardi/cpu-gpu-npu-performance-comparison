import argparse

from bench.runner import run_study


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_study(smoke=args.smoke)


if __name__ == "__main__":
    main()
