#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the semantic search model")
    subparsers.add_parser(
        "verify_embeddings", help="Verify the semantic search embeddings"
    )

    embed_parser = subparsers.add_parser(
        "embed_text", help="Generate embeddings for input text"
    )
    embed_parser.add_argument("text", type=str, help="Text to embed")

    embed_query_parser = subparsers.add_parser(
        "embedquery", help="Generate embeddings for input query"
    )
    embed_query_parser.add_argument("query", type=str, help="Query to embed")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
