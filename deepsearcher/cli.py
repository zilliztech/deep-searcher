import argparse
import logging
import sys
import warnings

from deepsearcher.configuration import Configuration, init_config
from deepsearcher.offline_loading import load_from_local_files, load_from_website
from deepsearcher.online_query import query
from deepsearcher.utils import log

httpx_logger = logging.getLogger("httpx")  # disable openai's logger output
httpx_logger.setLevel(logging.WARNING)


warnings.simplefilter(action="ignore", category=FutureWarning)  # disable warning output


def main():
    """
    Main entry point for the DeepSearcher CLI.

    This function parses command line arguments and executes the appropriate action
    based on the subcommand provided (query or load). It handles the deprecated
    command line format and provides helpful error messages.

    Returns:
        None
    """
    if "--query" in sys.argv or "--load" in sys.argv:
        print("\033[91m[Deprecated]\033[0m The use of '--query' and '--load' is deprecated.")
        print("Please use:")
        print("  deepsearcher query <your_query> --max_iter 3")
        print(
            "  deepsearcher load <your_local_path_or_url> --collection_name <your_collection_name> --collection_desc <your_collection_description>"
        )
        sys.exit(1)

    config = Configuration()  # Customize your config here
    init_config(config=config)

    parser = argparse.ArgumentParser(prog="deepsearcher", description="Deep Searcher.")
    subparsers = parser.add_subparsers(dest="subcommand", title="subcommands")

    ## Arguments of query
    query_parser = subparsers.add_parser("query", help="Query a question or search topic.")
    query_parser.add_argument("query", type=str, default="", help="query question or search topic.")
    query_parser.add_argument(
        "--max_iter",
        type=int,
        default=3,
        help="Max iterations of reflection. Default is 3.",
    )

    ## Arguments of loading
    load_parser = subparsers.add_parser(
        "load", help="Load knowledge from local files or from URLs."
    )
    load_parser.add_argument(
        "load_path",
        type=str,
        nargs="+",  # 1 or more files or urls
        help="Load knowledge from local files or from URLs.",
    )
    load_parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for loading knowledge.",
    )
    load_parser.add_argument(
        "--collection_name",
        type=str,
        default=None,
        help="Destination collection name of loaded knowledge.",
    )
    load_parser.add_argument(
        "--collection_desc",
        type=str,
        default=None,
        help="Description of the collection.",
    )
    load_parser.add_argument(
        "--force_new_collection",
        type=bool,
        default=False,
        help="If you want to drop origin collection and create a new collection on every load, set to True",
    )

    args = parser.parse_args()
    if args.subcommand == "query":
        final_answer, refs, consumed_tokens = query(args.query, max_iter=args.max_iter)
        log.color_print("\n### References\n")
        for i, ref in enumerate(refs):
            log.color_print(f"{i + 1}. {ref.text[:60]}… {ref.reference}")
    elif args.subcommand == "load":
        urls = [url for url in args.load_path if url.startswith("http")]
        local_files = [file for file in args.load_path if not file.startswith("http")]
        kwargs = {}
        if args.collection_name:
            kwargs["collection_name"] = args.collection_name
        if args.collection_desc:
            kwargs["collection_description"] = args.collection_desc
        if args.force_new_collection:
            kwargs["force_new_collection"] = args.force_new_collection
        if args.batch_size:
            kwargs["batch_size"] = args.batch_size
        if len(urls) > 0:
            load_from_website(urls, **kwargs)
        if len(local_files) > 0:
            load_from_local_files(local_files, **kwargs)
    else:
        print("Please provide a query or a load argument.")


if __name__ == "__main__":
    main()
