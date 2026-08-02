import argparse

from constants import CHUNK_SIZE, LIMIT, OVERLAP
from lib.semantic_search import (
    chunk,
    embed_chunks,
    embed_query_text,
    embed_text,
    search,
    search_chunked,
    semantic_chunk,
    verify_embeddings,
    verify_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help= "Verifying the LM model to be used for semantic search")

    embedtext_parser = subparsers.add_parser("embed_text", help= "Create embeddins for the input text")
    embedtext_parser.add_argument("text", type=str, help= "Input text")
        
    subparsers.add_parser("verify_embeddings", help= "Verify the embeddings for the movie dataset")
    
    embed_query_parser = subparsers.add_parser("embed_query", help= "Create embeddings for the input query")
    embed_query_parser.add_argument("query", type=str, help= "Input query")
    
    search_parser = subparsers.add_parser("search", help= "Search for the input query in the movie dataset")
    search_parser.add_argument("query", type=str, help= "Input query")
    search_parser.add_argument("--limit", type=int, default=LIMIT, help="Limit the number of results returned (default: %(default)s)")
    
    chunk_parser = subparsers.add_parser("chunk", help= "Chunk the input text into smaller pieces")
    chunk_parser.add_argument("text", type=str, help= "Input text")
    chunk_parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Size of each chunk (default: %(default)s)")
    chunk_parser.add_argument("--overlap", type=int, default=OVERLAP, help="Overlap between chunks (default: %(default)s)")
    
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help= "Chunk the input text into smaller pieces for semantic search")
    semantic_chunk_parser.add_argument("text", type=str, help= "Input text")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Size of each chunk (default: %(default)s)")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Overlap between chunks (default: %(default)s)")
    
    subparsers.add_parser("embed_chunks", help= "Create embeddings for the chunks of the movie dataset")

    search_chunked_parser = subparsers.add_parser("search_chunked", help= "Search for the input query in the chunked movie dataset")
    search_chunked_parser.add_argument("query", type=str, help= "Input query")
    search_chunked_parser.add_argument("--limit", type=int, nargs = "?", default=LIMIT, help="Limit the number of results returned (default: %(default)s)")
    
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            search(args.query, args.limit)
        case "chunk":
            chunk(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            chunks = semantic_chunk(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            for i, c in enumerate(chunks, start=1):
                print(f"{i}. {c}")
        case "embed_chunks":
            embed_chunks()
        case "search_chunked":
            results = search_chunked(args.query, args.limit)
            print(f"Query: {args.query}")
            print(f"Top {args.limit} results:")
            for i, result in enumerate(results, start=1):
                print(f"{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['document']}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()