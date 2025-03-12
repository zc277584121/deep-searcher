import os
from typing import List

from openai._types import NOT_GIVEN

from deepsearcher.embedding.base import BaseEmbedding

OPENAI_MODEL_DIM_MAP = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


class OpenAIEmbedding(BaseEmbedding):
    """
    OpenAI embedding model implementation.

    This class provides an interface to the OpenAI embedding API, which offers
    various embedding models for text processing.

    For more information, see:
    https://platform.openai.com/docs/guides/embeddings/use-cases
    """

    def __init__(self, model: str = "text-embedding-ada-002", **kwargs):
        """
        Initialize the OpenAI embedding model.

        Args:
            model (str): The model identifier to use for embeddings. Default is "text-embedding-ada-002".
            **kwargs: Additional keyword arguments.
                - api_key (str, optional): The OpenAI API key. If not provided,
                  it will be read from the OPENAI_API_KEY environment variable.
                - base_url (str, optional): The base URL for the OpenAI API. If not provided,
                  it will be read from the OPENAI_BASE_URL environment variable.
                - model_name (str, optional): Alternative way to specify the model.
                - dimension (int, optional): The dimension of the embedding vectors.
                  If not provided, the default dimension for the model will be used.

        Notes:
            Available models:
                - 'text-embedding-ada-002': No dimension needed, default is 1536
                - 'text-embedding-3-small': dimensions from 512 to 1536, default is 1536
                - 'text-embedding-3-large': dimensions from 1024 to 3072, default is 3072
        """
        from openai import OpenAI

        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = os.getenv("OPENAI_API_KEY")
        if "base_url" in kwargs:
            base_url = kwargs.pop("base_url")
        else:
            base_url = os.getenv("OPENAI_BASE_URL")
        if "model_name" in kwargs and (not model or model == "text-embedding-ada-002"):
            model = kwargs.pop("model_name")
        if "dimension" in kwargs:
            dimension = kwargs.pop("dimension")
        else:
            dimension = OPENAI_MODEL_DIM_MAP[model]
        self.dim = dimension
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url, **kwargs)

    def _get_dim(self):
        """
        Get the dimension parameter for the API call.

        Returns:
            int or NOT_GIVEN: The dimension to use for the embedding, or NOT_GIVEN
            if using text-embedding-ada-002 which doesn't support custom dimensions.
        """
        return self.dim if self.model != "text-embedding-ada-002" else NOT_GIVEN

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text (str): The query text to embed.

        Returns:
            List[float]: A list of floats representing the embedding vector.
        """
        # text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(
                input=[text], model=self.model, dimensions=self._get_dim()
            )
            .data[0]
            .embedding
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of document texts.

        Args:
            texts (List[str]): A list of document texts to embed.

        Returns:
            List[List[float]]: A list of embedding vectors, one for each input text.
        """
        res = self.client.embeddings.create(
            input=texts, model=self.model, dimensions=self._get_dim()
        )
        res = [r.embedding for r in res.data]
        return res

    @property
    def dimension(self) -> int:
        """
        Get the dimensionality of the embeddings for the current model.

        Returns:
            int: The number of dimensions in the embedding vectors.
        """
        return self.dim
