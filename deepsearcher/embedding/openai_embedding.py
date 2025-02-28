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
    https://platform.openai.com/docs/guides/embeddings/use-cases
    """

    def __init__(self, model: str = "text-embedding-ada-002", **kwargs):
        """

        Args:
            model_name (`str`):
                Can be one of the following:
                    'text-embedding-ada-002': No dimension needed, default is 1536,
                    'text-embedding-3-small': dimensions from 512 to 1536, default is 1536,
                    'text-embedding-3-large': dimensions from 1024 to 3072, default is 3072,
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
        return self.dim if self.model != "text-embedding-ada-002" else NOT_GIVEN

    def embed_query(self, text: str) -> List[float]:
        # text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(
                input=[text], model=self.model, dimensions=self._get_dim()
            )
            .data[0]
            .embedding
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        res = self.client.embeddings.create(
            input=texts, model=self.model, dimensions=self._get_dim()
        )
        res = [r.embedding for r in res.data]
        return res

    @property
    def dimension(self) -> int:
        return self.dim
