from typing import List, Optional, Set, Tuple

from deepsearcher.agent.base import BaseAgent
from deepsearcher.llm.base import BaseLLM
from deepsearcher.utils import log
from deepsearcher.vector_db.base import BaseVectorDB

COLLECTION_ROUTE_PROMPT = """
I provide you with collection_name(s) and corresponding collection_description(s). Please select the collection names that may be related to the question and return a python list of str. If there is no collection related to the question, you can return an empty list.

"QUESTION": {question}
"COLLECTION_INFO": {collection_info}

When you return, you can ONLY return a python list of str, WITHOUT any other additional content. Your selected collection name list is:
"""


class CollectionRouter(BaseAgent):
    """
    Routes queries to appropriate collections in the vector database.

    This class analyzes the content of a query and determines which collections
    in the vector database are most likely to contain relevant information.
    """

    def __init__(self, llm: BaseLLM, vector_db: BaseVectorDB, dim: int, **kwargs):
        """
        Initialize the CollectionRouter.

        Args:
            llm: The language model to use for analyzing queries.
            vector_db: The vector database containing the collections.
            dim: The dimension of the vector space to search in.
        """
        self.llm = llm
        self.vector_db = vector_db
        self.all_collections = [
            collection_info.collection_name
            for collection_info in self.vector_db.list_collections(dim=dim)
        ]

    def _get_authorized_collection_set(self, **kwargs) -> Optional[Set[str]]:
        authorized_collections = kwargs.get("authorized_collection_set")
        if authorized_collections is None:
            authorized_collections = kwargs.get("authorized_collections")
        if authorized_collections is None:
            return None
        return set(authorized_collections)

    def filter_authorized_collection_names(
        self, collection_names: List[str], **kwargs
    ) -> List[str]:
        authorized_collection_set = self._get_authorized_collection_set(**kwargs)
        if authorized_collection_set is None:
            return collection_names

        return [
            collection_name
            for collection_name in collection_names
            if collection_name in authorized_collection_set
        ]

    def _filter_authorized_collections(self, collection_infos: List, **kwargs) -> List:
        authorized_collection_set = self._get_authorized_collection_set(**kwargs)
        if authorized_collection_set is None:
            return collection_infos

        return [
            collection_info
            for collection_info in collection_infos
            if collection_info.collection_name in authorized_collection_set
        ]

    def invoke(self, query: str, dim: int, **kwargs) -> Tuple[List[str], int]:
        """
        Determine which collections are relevant for the given query.

        This method analyzes the query content and selects collections that are
        most likely to contain information relevant to answering the query.

        Args:
            query (str): The query to analyze.
            dim (int): The dimension of the vector space to search in.

        Returns:
            Tuple[List[str], int]: A tuple containing:
                - A list of selected collection names
                - The token usage for the routing operation
        """
        consume_tokens = 0
        collection_infos = self._filter_authorized_collections(
            self.vector_db.list_collections(dim=dim), **kwargs
        )
        if len(collection_infos) == 0:
            log.warning(
                "No collections found in the vector database. Please check the database connection."
            )
            return [], 0
        if len(collection_infos) == 1:
            the_only_collection = collection_infos[0].collection_name
            log.color_print(
                f"<think> Perform search [{query}] on the vector DB collection: {the_only_collection} </think>\n"
            )
            return [the_only_collection], 0
        vector_db_search_prompt = COLLECTION_ROUTE_PROMPT.format(
            question=query,
            collection_info=[
                {
                    "collection_name": collection_info.collection_name,
                    "collection_description": collection_info.description,
                }
                for collection_info in collection_infos
            ],
        )
        chat_response = self.llm.chat(
            messages=[{"role": "user", "content": vector_db_search_prompt}]
        )
        selected_collections = self.llm.literal_eval(chat_response.content)
        consume_tokens += chat_response.total_tokens
        allowed_collection_names = {
            collection_info.collection_name for collection_info in collection_infos
        }
        selected_collections = [
            collection_name
            for collection_name in selected_collections
            if collection_name in allowed_collection_names
        ]

        for collection_info in collection_infos:
            # If a collection description is not provided, use the query as the search query
            if not collection_info.description:
                selected_collections.append(collection_info.collection_name)
            # If the default collection exists, use the query as the search query
            if self.vector_db.default_collection == collection_info.collection_name:
                selected_collections.append(collection_info.collection_name)
        selected_collections = list(set(selected_collections))
        log.color_print(
            f"<think> Perform search [{query}] on the vector DB collections: {selected_collections} </think>\n"
        )
        return selected_collections, consume_tokens
