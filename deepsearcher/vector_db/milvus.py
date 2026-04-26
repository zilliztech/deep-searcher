from typing import List, Optional, Union

import numpy as np
from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker

from deepsearcher.loader.splitter import Chunk
from deepsearcher.utils import log
from deepsearcher.vector_db.base import BaseVectorDB, CollectionInfo, RetrievalResult


class Milvus(BaseVectorDB):
    """Milvus class is a subclass of DB class."""

    client: MilvusClient = None

    def __init__(
        self,
        default_collection: str = "deepsearcher",
        uri: str = "http://localhost:19530",
        token: str = "root:Milvus",
        user: str = "",
        password: str = "",
        db: str = "default",
        hybrid: bool = False,
        **kwargs,
    ):
        """
        Initialize the Milvus client.

        Args:
            default_collection (str, optional): Default collection name. Defaults to "deepsearcher".
            uri (str, optional): URI for connecting to Milvus server. Defaults to "http://localhost:19530".
            token (str, optional): Authentication token for Milvus. Defaults to "root:Milvus".
            user (str, optional): Username for authentication. Defaults to "".
            password (str, optional): Password for authentication. Defaults to "".
            db (str, optional): Database name. Defaults to "default".
            hybrid (bool, optional): Whether to enable hybrid search. Defaults to False.
            **kwargs: Additional keyword arguments to pass to the MilvusClient.
        """
        super().__init__(default_collection)
        self.default_collection = default_collection
        self.client = MilvusClient(
            uri=uri, user=user, password=password, token=token, db_name=db, timeout=30, **kwargs
        )

        self.hybrid = hybrid

    def init_collection(
        self,
        dim: int,
        collection: Optional[str] = "deepsearcher",
        description: Optional[str] = "",
        force_new_collection: bool = False,
        text_max_length: int = 65_535,
        reference_max_length: int = 2048,
        metric_type: str = "L2",
        *args,
        **kwargs,
    ):
        """
        Initialize a collection in Milvus.

        Args:
            dim (int): Dimension of the vector embeddings.
            collection (Optional[str], optional): Collection name. Defaults to "deepsearcher".
            description (Optional[str], optional): Collection description. Defaults to "".
            force_new_collection (bool, optional): Whether to force create a new collection if it already exists. Defaults to False.
            text_max_length (int, optional): Maximum length for text field. Defaults to 65_535.
            reference_max_length (int, optional): Maximum length for reference field. Defaults to 2048.
            metric_type (str, optional): Metric type for vector similarity search. Defaults to "L2".
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        if not collection:
            collection = self.default_collection
        if description is None:
            description = ""

        self.metric_type = metric_type

        try:
            has_collection = self.client.has_collection(collection, timeout=5)
            if force_new_collection and has_collection:
                self.client.drop_collection(collection)
            elif has_collection:
                return
            schema = self.client.create_schema(
                enable_dynamic_field=False, auto_id=True, description=description
            )
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)

            if self.hybrid:
                analyzer_params = {"tokenizer": "standard", "filter": ["lowercase"]}
                schema.add_field(
                    "text",
                    DataType.VARCHAR,
                    max_length=text_max_length,
                    analyzer_params=analyzer_params,
                    enable_match=True,
                    enable_analyzer=True,
                )
            else:
                schema.add_field("text", DataType.VARCHAR, max_length=text_max_length)

            schema.add_field("reference", DataType.VARCHAR, max_length=reference_max_length)
            schema.add_field("metadata", DataType.JSON)

            if self.hybrid:
                schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
                bm25_function = Function(
                    name="bm25",
                    function_type=FunctionType.BM25,
                    input_field_names=["text"],
                    output_field_names="sparse_vector",
                )
                schema.add_function(bm25_function)

            index_params = self.client.prepare_index_params()
            index_params.add_index(field_name="embedding", metric_type=metric_type)

            if self.hybrid:
                index_params.add_index(
                    field_name="sparse_vector",
                    index_type="SPARSE_INVERTED_INDEX",
                    metric_type="BM25",
                )

            self.client.create_collection(
                collection,
                schema=schema,
                index_params=index_params,
                consistency_level="Strong",
            )
            log.color_print(f"create collection [{collection}] successfully")
        except Exception as e:
            log.critical(f"fail to init db for milvus, error info: {e}")

    def insert_data(
        self,
        collection: Optional[str],
        chunks: List[Chunk],
        batch_size: int = 256,
        *args,
        **kwargs,
    ):
        """
        Insert data into a Milvus collection.

        Args:
            collection (Optional[str]): Collection name. If None, uses default_collection.
            chunks (List[Chunk]): List of Chunk objects to insert.
            batch_size (int, optional): Number of chunks to insert in each batch. Defaults to 256.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        if not collection:
            collection = self.default_collection
        texts = [chunk.text for chunk in chunks]
        references = [chunk.reference for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks]

        datas = [
            {
                "embedding": embedding,
                "text": text,
                "reference": reference,
                "metadata": metadata,
            }
            for embedding, text, reference, metadata in zip(
                embeddings, texts, references, metadatas
            )
        ]
        batch_datas = [datas[i : i + batch_size] for i in range(0, len(datas), batch_size)]
        try:
            for batch_data in batch_datas:
                self.client.insert(collection_name=collection, data=batch_data)
        except Exception as e:
            log.critical(f"fail to insert data, error info: {e}")

    def search_data(
        self,
        collection: Optional[str],
        vector: Union[np.array, List[float]],
        top_k: int = 5,
        query_text: Optional[str] = None,
        *args,
        **kwargs,
    ) -> List[RetrievalResult]:
        """
        Search for similar vectors in a Milvus collection.

        Args:
            collection (Optional[str]): Collection name. If None, uses default_collection.
            vector (Union[np.array, List[float]]): Query vector for similarity search.
            top_k (int, optional): Number of results to return. Defaults to 5.
            query_text (Optional[str], optional): Original query text for hybrid search. Defaults to None.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            List[RetrievalResult]: List of retrieval results containing similar vectors.
        """
        if not collection:
            collection = self.default_collection
        try:
            use_hybrid = self.hybrid and query_text

            if use_hybrid:
                sparse_search_params = {"metric_type": "BM25"}
                sparse_request = AnnSearchRequest(
                    [query_text], "sparse_vector", sparse_search_params, limit=top_k
                )

                dense_search_params = {"metric_type": self.metric_type}
                dense_request = AnnSearchRequest(
                    [vector], "embedding", dense_search_params, limit=top_k
                )

                search_results = self.client.hybrid_search(
                    collection_name=collection,
                    reqs=[sparse_request, dense_request],
                    ranker=RRFRanker(),
                    limit=top_k,
                    output_fields=["text", "reference", "metadata"],
                    timeout=10,
                )
            else:
                search_results = self.client.search(
                    collection_name=collection,
                    data=[vector],
                    limit=top_k,
                    output_fields=["text", "reference", "metadata"],
                    timeout=10,
                )

            return [
                RetrievalResult(
                    embedding=None,
                    text=b["entity"]["text"],
                    reference=b["entity"]["reference"],
                    score=b["distance"],
                    metadata=b["entity"]["metadata"],
                )
                for a in search_results
                for b in a
            ]
        except Exception as e:
            log.critical(f"fail to search data, error info: {e}")
            return []

    def list_collections(self, *args, **kwargs) -> List[CollectionInfo]:
        """
        List all collections in the Milvus database.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            List[CollectionInfo]: List of collection information objects.
        """
        collection_infos = []
        dim = kwargs.pop("dim", 0)
        try:
            collections = self.client.list_collections()
            for collection in collections:
                description = self.client.describe_collection(collection)
                if dim != 0:
                    skip = False
                    for field_dict in description["fields"]:
                        if (
                            field_dict["name"] == "embedding"
                            and field_dict["type"] == DataType.FLOAT_VECTOR
                        ):
                            if field_dict["params"]["dim"] != dim:
                                skip = True
                    if skip:
                        continue
                collection_infos.append(
                    CollectionInfo(
                        collection_name=collection,
                        description=description["description"],
                    )
                )
        except Exception as e:
            log.critical(f"fail to list collections, error info: {e}")
        return collection_infos

    def clear_db(self, collection: str = "deepsearcher", *args, **kwargs):
        """
        Clear (drop) a collection from the Milvus database.

        Args:
            collection (str, optional): Collection name to drop. Defaults to "deepsearcher".
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        if not collection:
            collection = self.default_collection
        try:
            self.client.drop_collection(collection)
        except Exception as e:
            log.warning(f"fail to clear db, error info: {e}")
