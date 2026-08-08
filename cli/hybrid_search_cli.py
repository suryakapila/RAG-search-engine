import argparse
import time

from lib.hybrid_search import normalize, weighted_search, rrf_search
from lib.query_enhance import enhance_query, rerank_batch, rerank_rrf, rerank_cross_encoder


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
    rrf_search_parser.add_argument("--enhance", type=str,choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Re-ranking the rrf search output")
    rrf_search_parser.add_argument("--rerank-pool", type=int, default=5, help="Candidate pool multiplier for reranking (pool size = multiplier * limit)")
    rrf_search_parser.add_argument("--rerank-sleep", type=float, default=3.0, help="Seconds to sleep between rerank LLM calls")
    
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
            query = args.query
            if args.enhance:
                enhanced = enhance_query(query, args.enhance)
                print(f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced}'\n")
                query = enhanced

            results = rrf_search(query, args.k, args.limit)

            if args.rerank_method:
                candidates = results[: args.rerank_pool * args.limit]
                if args.rerank_method == "individual":
                    for result in candidates:
                        result["rerank_score"] = rerank_rrf(args.query, result)
                        time.sleep(args.rerank_sleep)
                    candidates.sort(key=lambda d: d["rerank_score"], reverse=True)
                    res = candidates[: args.limit]
                elif args.rerank_method == "batch":
                    ranked_ids = rerank_batch(args.query, candidates)
                    rank_of = {doc_id: rank for rank, doc_id in enumerate(ranked_ids, start=1)}
                    for result in candidates:
                        result["rerank_rank"] = rank_of[result["id"]]
                    candidates.sort(key=lambda d: d["rerank_rank"])
                    res = candidates[: args.limit]
                elif args.rerank_method =="cross_encoder":
                    scores = rerank_cross_encoder(args.query, candidates)
                    for i in range(0, len(candidates)):
                        candidates[i]["cross_encoder_score"] = scores[i]
                    candidates.sort(key = lambda d: d["cross_encoder_score"], reverse= True)
                    res = candidates[:args.limit]
            else:
                res = results[: args.limit]

            for i, r in enumerate(res, start=1):
                bm25_rank = r["bm25_rank"] if r["bm25_rank"] is not None else "-"
                sem_rank = r["sem_rank"] if r["sem_rank"] is not None else "-"
                print(f"{i}. {r['title']}")
                if args.rerank_method == 'individual':
                    print(f"  Re-rank Score: {r['rerank_score']:.3f}")
                elif args.rerank_method == 'batch':
                    print(f"  Re-rank Rank: {r['rerank_rank']}")
                elif args.rerank_method == 'cross_encoder':
                    print(f"  Cross Encoder Score: {r['cross_encoder_score']:.3f}")
                print(f"  RRF Score: {r['rrf_score']:.3f}")
                print(f"  BM25 Rank: {bm25_rank}, Semantic Rank: {sem_rank}")
                print(f"  {r['document']}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
