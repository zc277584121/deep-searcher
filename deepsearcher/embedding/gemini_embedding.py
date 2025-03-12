import os
from typing import List

from deepsearcher.embedding.base import BaseEmbedding

GEMINI_MODEL_DIM_MAP = {
    "text-embedding-004": 768,
    "gemini-embedding-exp-03-07": 3072,
}


class GeminiEmbedding(BaseEmbedding):
    """
    https://ai.google.dev/api/embeddings
    """

    def __init__(self, model: str = "text-embedding-004", **kwargs):
        """

        Args:
            model_name (`str`):
                Can be one of the following:
                    'text-embedding-004': dimemsions from 1 to 768, default is 768
                    'gemini-embedding-exp-03-07': dimensions from 1 to 3072, default is 3072
        """
        from google import genai

        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = os.getenv("GEMINI_API_KEY")
        if "dimension" in kwargs:
            dimension = kwargs.pop("dimension")
        else:
            dimension = GEMINI_MODEL_DIM_MAP[model]
        self.dim = dimension
        self.model = model
        self.client = genai.Client(api_key=api_key, **kwargs)

    def _get_dim(self):
        return self.dim

    def _embed_content(self, texts: List[str]):
        from google.genai import types

        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=self._get_dim()),
        )
        return response.embeddings

    def embed_query(self, text: str) -> List[float]:
        # make sure the text is one string
        if len(text) != 1:
            text = " ".join(text)
        result = self._embed_content(text)
        # embedding = [r.values for r in result]
        embedding = result[0].values
        return embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # For Gemini free level, the maximum rqeusts in one batch is 100, so we need to split the texts again
        sub_splits = [texts[i : i + 100] for i in range(0, len(texts), 100)]
        embeddings = []
        for texts in sub_splits:
            result = self._embed_content(texts)
            embeddings.extend([r.values for r in result])
        return embeddings

    @property
    def dimension(self) -> int:
        return self.dim
