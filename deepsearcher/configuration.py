from typing import Literal

from deepsearcher.embedding.base import BaseEmbedding
from deepsearcher.llm.base import BaseLLM
from deepsearcher.loader.file_loader.base import BaseLoader
from deepsearcher.loader.web_crawler.base import BaseCrawler
from deepsearcher.vector_db.base import BaseVectorDB

FeatureType = Literal["llm", "embedding", "file_loader", "web_crawler", "vector_db"]

class Configuration:
    def __init__(self):
        # Initialize default configurations
        self.provide_settings = {
            "llm": {
                "provider": "OpenAI",#"TogetherAI",
                "config": {
                    "model": "gpt-4o-mini" #"gpt-4o"#"deepseek-ai/DeepSeek-V3"
                }
            },
            "embedding": {
                "provider": "MilvusEmbedding",
                "config": {
                    "model_name": "default"
                }
            },
            "file_loader": {
                "provider": "PDFLoader",
                "config": {}
            },
            "web_crawler": {
                "provider": "FireCrawlCrawler",
                "config": {}
            },
            "vector_db": {
                "provider": "Milvus",
                "config": {
                    "uri": "./milvus.db"
                }
            }
        }
        self.query_settings = {
            "max_iter": 3
        }
        self.load_settings = {
            "chunk_size": 1500,
            "chunk_overlap": 100
        }

    def set_provider_config(self, feature: FeatureType, provider: str, provider_configs: dict):
        """
        Set the provider and its configurations for a given feature.

        :param feature: The feature to configure (e.g., 'llm', 'file_loader', 'web_crawler').
        :param provider: The provider name (e.g., 'openai', 'deepseek').
        :param provider_configs: A dictionary with configurations specific to the provider.
        """
        if feature not in self.provide_settings:
            raise ValueError(f"Unsupported feature: {feature}")

        self.provide_settings[feature]["provider"] = provider
        self.provide_settings[feature]["config"] = provider_configs


    def get_provider_config(self, feature: FeatureType):
        """
        Get the current provider and configuration for a given feature.

        :param feature: The feature to retrieve (e.g., 'llm', 'file_loader', 'web_crawler').
        :return: A dictionary with provider and its configurations.
        """
        if feature not in self.provide_settings:
            raise ValueError(f"Unsupported feature: {feature}")

        return self.provide_settings[feature]

class ModuleFactory:
    def __init__(self, config: Configuration):
        self.config = config
    
    def _create_module_instance(self, feature: FeatureType, module_name: str):
        # e.g.
        # feature = "file_loader"
        # module_name = "deepsearcher.loader.file_loader"
        class_name = self.config.provide_settings[feature]["provider"]
        module = __import__(module_name, fromlist=[class_name])
        class_ = getattr(module, class_name)
        return class_(**self.config.provide_settings[feature]["config"])
    
    def create_llm(self) -> BaseLLM:
        return self._create_module_instance("llm", "deepsearcher.llm")
    
    def create_embedding(self) -> BaseEmbedding:
        return self._create_module_instance("embedding", "deepsearcher.embedding")
    
    def create_file_loader(self) -> BaseLoader:
        return self._create_module_instance("file_loader", "deepsearcher.loader.file_loader")
    
    def create_web_crawler(self) -> BaseCrawler:
        return self._create_module_instance("web_crawler", "deepsearcher.loader.web_crawler")
    
    def create_vector_db(self) -> BaseVectorDB:
        return self._create_module_instance("vector_db", "deepsearcher.vector_db")
    

config = Configuration()

module_factory: ModuleFactory = None
llm: BaseLLM = None
embedding_model: BaseEmbedding = None
file_loader: BaseLoader = None
vector_db: BaseVectorDB = None
web_crawler: BaseCrawler = None


def init_config(config: Configuration):
    global module_factory, llm, embedding_model, file_loader, vector_db, web_crawler
    module_factory = ModuleFactory(config)
    llm = module_factory.create_llm()
    embedding_model = module_factory.create_embedding()
    file_loader = module_factory.create_file_loader()
    web_crawler = module_factory.create_web_crawler()
    vector_db = module_factory.create_vector_db()