import argparse
from lib.multimodal_search import verify_image_embedding, image_search_command


def main():
    parser = argparse.ArgumentParser(description="Verify image embedding")
    sub_parser = parser.add_subparsers(dest="command")
    image_parser = sub_parser.add_parser(
        "verify_image_embedding", help="Verify image embedding"
    )
    image_parser.add_argument("image_path", help="Path to the image file")

    image_search_parser = sub_parser.add_parser(
        "image_search", help="Search for movies using an image"
    )
    image_search_parser.add_argument("image_path", help="Path to the image file")
    image_search_parser.add_argument(
        "--top-k", type=int, default=5, help="Number of results to return"
    )
    args = parser.parse_args()

    if args.command == "verify_image_embedding":
        verify_image_embedding(args.image_path)
    elif args.command == "image_search":
        results = image_search_command(args.image_path, args.top_k)
        for i, res in enumerate(results):
            print(f"{i + 1}. {res['title']} (similarity: {res['score']:.3f})")
            print(f"   {res['description']}")
            print()


if __name__ == "__main__":
    main()
