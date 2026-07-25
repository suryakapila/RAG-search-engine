import argparse
from utils import tokenise_query, preprocess_text, has_matching_tokens, filter_stopwords, stemmed_tokens, load_movies, build_command, InvertedIndex, tf, idf, tfidf
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="build inverted index for movies")

    tf_parser = subparsers.add_parser("tf", help= "term frequency in a document for movies")
    tf_parser.add_argument("doc_id", type= int, help ="document id of movie")
    tf_parser.add_argument("term", type= str, help = "term to find the term frequency")

    idf_parser = subparsers.add_parser("idf", help= "Inverse Document Frequency for better search")
    idf_parser.add_argument("term", type=str, help="search term")

    tfidf_parser = subparsers.add_parser("tfidf", help= "tfidf gives one relevance score")
    tfidf_parser.add_argument("doc_id", type=int, help="document id ")
    tfidf_parser.add_argument("term", type=str, help= "query term")

    args = parser.parse_args()

    match args.command:
        case "search":
            query_tokens = tokenise_query(args.query)
            cleaned_query_tokens = filter_stopwords(query_tokens)
            final_query_tokens = stemmed_tokens(cleaned_query_tokens)

            print(f"Searching for: {args.query}")
            try:
                index = InvertedIndex()
                index.load()
            except FileNotFoundError:
                print("file not found")
                sys.exit(1)
            res = set()
            done = False
            for token in final_query_tokens:
                ids = index.get_documents(token)
                for id in ids:
                    if len(res)< 5:
                        res.add(id)
                    if len(res) == 5:
                        done = True
                        break
                if done:
                    break
            for number, id in enumerate(res, start = 1):
                movie = index.docmap[id]
                print(f"{number}: {movie['title']}")
                    
        case "build":
            build_command()
        case "tf":
            tf_score = tf(args.doc_id, args.term)
            print(f"{tf_score}")

        case "idf":
            idf_score = idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf_score:.2f}")

        case "tfidf":
            tf_idf = tfidf(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()