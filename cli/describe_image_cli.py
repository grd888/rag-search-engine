import argparse
from lib.multimodal_rewrite import describe_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    parser.add_argument("--image", type=str, help="Path of image to describe")
    parser.add_argument("--query", type=str, help="Query to use for description")

    args = parser.parse_args()
    describe_command(args.image, args.query)

if __name__ == "__main__":
    main()