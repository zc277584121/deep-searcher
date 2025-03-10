import argparse
from typing import Dict, List, Union

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from deepsearcher.configuration import Configuration, init_config
from deepsearcher.offline_loading import load_from_local_files, load_from_website
from deepsearcher.online_query import query

app = FastAPI()

config = Configuration()

init_config(config)


class ProviderConfigRequest(BaseModel):
    feature: str
    provider: str
    config: Dict


@app.post("/set-provider-config/")
def set_provider_config(request: ProviderConfigRequest):
    try:
        config.set_provider_config(request.feature, request.provider, request.config)
        init_config(config)
        return {
            "message": "Provider config set successfully",
            "provider": request.provider,
            "config": request.config,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set provider config: {str(e)}")


@app.post("/load-files/")
def load_files(
    paths: Union[str, List[str]] = Body(
        ...,
        description="A list of file paths to be loaded.",
        examples=["/path/to/file1", "/path/to/file2", "/path/to/dir1"],
    ),
    collection_name: str = Body(
        None,
        description="Optional name for the collection.",
        examples=["my_collection"],
    ),
    collection_description: str = Body(
        None,
        description="Optional description for the collection.",
        examples=["This is a test collection."],
    ),
    batch_size: int = Body(
        None,
        description="Optional batch size for the collection.",
        examples=[256],
    ),
):
    try:
        load_from_local_files(
            paths_or_directory=paths,
            collection_name=collection_name,
            collection_description=collection_description,
            batch_size=batch_size,
        )
        return {"message": "Files loaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load-website/")
def load_website(
    urls: Union[str, List[str]] = Body(
        ...,
        description="A list of URLs of websites to be loaded.",
        examples=["https://milvus.io/docs/overview.md"],
    ),
    collection_name: str = Body(
        None,
        description="Optional name for the collection.",
        examples=["my_collection"],
    ),
    collection_description: str = Body(
        None,
        description="Optional description for the collection.",
        examples=["This is a test collection."],
    ),
    batch_size: int = Body(
        None,
        description="Optional batch size for the collection.",
        examples=[256],
    ),
):
    try:
        load_from_website(
            urls=urls,
            collection_name=collection_name,
            collection_description=collection_description,
            batch_size=batch_size,
        )
        return {"message": "Website loaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query/")
def perform_query(
    original_query: str = Query(
        ...,
        description="Your question here.",
        examples=["Write a report about Milvus."],
    ),
    max_iter: int = Query(
        3,
        description="The maximum number of iterations for reflection.",
        ge=1,
        examples=[3],
    ),
):
    try:
        result_text, _, consume_token = query(original_query, max_iter)
        return {"result": result_text, "consume_token": consume_token}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastAPI Server")
    parser.add_argument("--enable-cors", type=bool, default=False, help="Enable CORS support")
    args = parser.parse_args()
    if args.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        print("CORS is enabled.")
    else:
        print("CORS is disabled.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
