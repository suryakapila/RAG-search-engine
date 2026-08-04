import argparse

from lib.hybrid_search import normalize, weighted_search, rrf_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Print min-max normalized scores")
    normalize_parser.add_argument("scores", nargs="*", type=float, help="Input list of scores")
    
    weighted_search_parser = subparsers.add_parser("weighted-search", help= "WEIGHTED SCORE to determine the ranking of the search results")
    weighted_search_parser.add_argument("query", type= str, help = "Input query")
    weighted_search_parser.add_argument("--alpha", nargs = '?',  type = float, default = 0.5, help = "configurable alpha values")
    weighted_search_parser.add_argument("--limit", nargs= "?", type = int, default = 5, help = "configurable limit value")
    
    rrf_search_parser = subparsers.add_parser("rrf-search", help= "Build hybrid search using Reciprocal Rank Fusion.")
    rrf_search_parser.add_argument("query", type = str, help="Input query")
    rrf_search_parser.add_argument("-k", type= int, nargs = "?", default = 60, help= "k parameter (a constant) controls how much more weight we give to higher-ranked results vs. lower-ranked ones.")
    rrf_search_parser.add_argument("--limit", type= int, nargs="?", default= 5, help= "configurable limit value")
    
    args = parser.parse_args()

    match args.command:
        case "normalize":
            for score in normalize(args.scores):
                print(f"* {score:.4f}")
        case "weighted-search":
            results = weighted_search(args.query, args.alpha, args.limit)
            for i, r in enumerate(results, start=1):
                print(f"{i}. {r['title']}")
                print(f"  Hybrid Score: {r['hybrid_score']:.3f}")
                print(f"  BM25: {r['keyword_score']:.3f}, Semantic: {r['semantic_score']:.3f}")
                print(f"  {r['document']}...")
        case "rrf-search":
            results = rrf_search(args.query, args.k, args.limit)
            for i, r in enumerate(results, start=1):
                bm25_rank = r["bm25_rank"] if r["bm25_rank"] is not None else "-"
                sem_rank = r["sem_rank"] if r["sem_rank"] is not None else "-"
                print(f"{i}. {r['title']}")
                print(f"  RRF Score: {r['rrf_score']:.3f}")
                print(f"  BM25 Rank: {bm25_rank}, Semantic Rank: {sem_rank}")
                print(f"  {r['document']}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
