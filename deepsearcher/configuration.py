import os
from typing import Literal

import yaml

from deepsearcher.agent import ChainOfRAG, DeepSearch, NaiveRAG
from deepsearcher.agent.rag_router import RAGRouter
from deepsearcher.embedding.base import BaseEmbedding
from deepsearcher.llm.base import BaseLLM
from deepsearcher.loader.file_loader.base import BaseLoader
from deepsearcher.loader.web_crawler.base import BaseCrawler
from deepsearcher.vector_db.base import BaseVectorDB

current_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_YAML_PATH = os.path.join(current_dir, "config.yaml")

FeatureType = Literal["llm", "embedding", "file_loader", "web_crawler", "vector_db"]


class Configuration:
    """
    Configuration class for DeepSearcher.

    This class manages the configuration settings for various components of the DeepSearcher system,
    including LLM providers, embedding models, file loaders, web crawlers, and vector databases.
    It loads configurations from a YAML file and provides methods to get and set provider configurations.
    """

    def __init__(self, config_path: str = DEFAULT_CONFIG_YAML_PATH):
        """
        Initialize the Configuration object.

        Args:
            config_path: Path to the configuration YAML file. Defaults to the config.yaml in the project root.
        """
        # Initialize default configurations
        config_data = self.load_config_from_yaml(config_path)
        self.provide_settings = config_data["provide_settings"]
        self.query_settings = config_data["query_settings"]
        self.load_settings = config_data["load_settings"]

    def load_config_from_yaml(self, config_path: str):
        """
        Load configuration from a YAML file.

        Args:
            config_path: Path to the configuration YAML file.

        Returns:
            The loaded configuration data as a dictionary.
        """
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def set_provider_config(self, feature: FeatureType, provider: str, provider_configs: dict):
        """
        Set the provider and its configurations for a given feature.

        Args:
            feature: The feature to configure (e.g., 'llm', 'file_loader', 'web_crawler').
            provider: The provider name (e.g., 'openai', 'deepseek').
            provider_configs: A dictionary with configurations specific to the provider.

        Raises:
            ValueError: If the feature is not supported.
        """
        if feature not in self.provide_settings:
            raise ValueError(f"Unsupported feature: {feature}")

        self.provide_settings[feature]["provider"] = provider
        self.provide_settings[feature]["config"] = provider_configs

    def get_provider_config(self, feature: FeatureType):
        """
        Get the current provider and configuration for a given feature.

        Args:
            feature: The feature to retrieve (e.g., 'llm', 'file_loader', 'web_crawler').

        Returns:
            A dictionary with provider and its configurations.

        Raises:
            ValueError: If the feature is not supported.
        """
        if feature not in self.provide_settings:
            raise ValueError(f"Unsupported feature: {feature}")

        return self.provide_settings[feature]


class ModuleFactory:
    """
    Factory class for creating instances of various modules in the DeepSearcher system.

    This class creates instances of LLMs, embedding models, file loaders, web crawlers,
    and vector databases based on the configuration settings.
    """

    def __init__(self, config: Configuration):
        """
        Initialize the ModuleFactory.

        Args:
            config: The Configuration object containing provider settings.
        """
        self.config = config

    def _create_module_instance(self, feature: FeatureType, module_name: str):
        """
        Create an instance of a module based on the feature and module name.

        Args:
            feature: The feature type (e.g., 'llm', 'embedding').
            module_name: The module name to import from.

        Returns:
            An instance of the specified module.
        """
        # e.g.
        # feature = "file_loader"
        # module_name = "deepsearcher.loader.file_loader"
        class_name = self.config.provide_settings[feature]["provider"]
        module = __import__(module_name, fromlist=[class_name])
        class_ = getattr(module, class_name)
        return class_(**self.config.provide_settings[feature]["config"])

    def create_llm(self) -> BaseLLM:
        """
        Create an instance of a language model.

        Returns:
            An instance of a BaseLLM implementation.
        """
        return self._create_module_instance("llm", "deepsearcher.llm")

    def create_embedding(self) -> BaseEmbedding:
        """
        Create an instance of an embedding model.

        Returns:
            An instance of a BaseEmbedding implementation.
        """
        return self._create_module_instance("embedding", "deepsearcher.embedding")

    def create_file_loader(self) -> BaseLoader:
        """
        Create an instance of a file loader.

        Returns:
            An instance of a BaseLoader implementation.
        """
        return self._create_module_instance("file_loader", "deepsearcher.loader.file_loader")

    def create_web_crawler(self) -> BaseCrawler:
        """
        Create an instance of a web crawler.

        Returns:
            An instance of a BaseCrawler implementation.
        """
        return self._create_module_instance("web_crawler", "deepsearcher.loader.web_crawler")

    def create_vector_db(self) -> BaseVectorDB:
        """
        Create an instance of a vector database.

        Returns:
            An instance of a BaseVectorDB implementation.
        """
        return self._create_module_instance("vector_db", "deepsearcher.vector_db")


module_factory: ModuleFactory = None
llm: BaseLLM = None
embedding_model: BaseEmbedding = None
file_loader: BaseLoader = None
vector_db: BaseVectorDB = None
web_crawler: BaseCrawler = None
default_searcher: RAGRouter = None
naive_rag: NaiveRAG = None


def init_config(config: Configuration):
    """
    Initialize the global configuration and create instances of all required modules.

    This function initializes the global variables for the LLM, embedding model,
    file loader, web crawler, vector database, and RAG agents.

    Args:
        config: The Configuration object to use for initialization.
    """
    global \
        module_factory, \
        llm, \
        embedding_model, \
        file_loader, \
        vector_db, \
        web_crawler, \
        default_searcher, \
        naive_rag
    module_factory = ModuleFactory(config)
    llm = module_factory.create_llm()
    embedding_model = module_factory.create_embedding()
    file_loader = module_factory.create_file_loader()
    web_crawler = module_factory.create_web_crawler()
    vector_db = module_factory.create_vector_db()

    default_searcher = RAGRouter(
        llm=llm,
        rag_agents=[
            DeepSearch(
                llm=llm,
                embedding_model=embedding_model,
                vector_db=vector_db,
                max_iter=config.query_settings["max_iter"],
                route_collection=True,
                text_window_splitter=True,
            ),
            ChainOfRAG(
                llm=llm,
                embedding_model=embedding_model,
                vector_db=vector_db,
                max_iter=config.query_settings["max_iter"],
                route_collection=True,
                text_window_splitter=True,
            ),
        ],
    )
    naive_rag = NaiveRAG(
        llm=llm,
        embedding_model=embedding_model,
        vector_db=vector_db,
        top_k=10,
        route_collection=True,
        text_window_splitter=True,
    )
