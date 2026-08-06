import argparse

from inferlab.runtime.runner import InferenceRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="inferlab")
    parser.add_argument("--model", required=True, help="Model name or path to load")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Hello from inferlab! Loading model: {args.model}")

    runner = InferenceRunner(args.model)

    while True:
        prompt = input("> ")
        response = runner.generate(prompt)
        print(response)


if __name__ == "__main__":
    main()
