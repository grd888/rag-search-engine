#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    search_command,
    chunk_text,
    semantick_chunk,
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

    search_parser = subparsers.add_parser("search", help="Search for movies")
    search_parser.add_argument("query", type=str, help="Query to search for")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk text")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Chunk size")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Overlap between chunks")
    
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Semantic chunk text")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to semantic chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Maximum chunk size")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Overlap between chunks")
    
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
        case "search":
            search_command(args.query, args.limit)
        case "chunk":
            print(f"Chunking {len(args.text)} characters")
            chunks = chunk_text(args.text, args.chunk_size, args.overlap)
            for i, chunk in enumerate(chunks):
                print(f"{i+1}. {chunk}")
        case "semantic_chunk":
            print(f"Semantically chunking {len(args.text)} characters")
            chunks = semantick_chunk(args.text, args.max_chunk_size, args.overlap)
            for i, chunk in enumerate(chunks):
                print(f"{i+1}. {chunk}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
