import argparse
from lib.multimodal_search import verify_image_embedding

def main():
    parser = argparse.ArgumentParser(description="Verify image embedding")
    sub_parser = parser.add_subparsers(dest="command")
    image_parser = sub_parser.add_parser("verify_image_embedding", help="Verify image embedding")
    image_parser.add_argument("image_path", help="Path to the image file")
    args = parser.parse_args()
    
    verify_image_embedding(args.image_path)

if __name__ == "__main__":
    main()