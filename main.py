from typing import List, Union

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query

from deepsearcher.configuration import Configuration, init_config
from deepsearcher.offline_loading import load_from_local_files, load_from_website
from deepsearcher.online_query import query

app = FastAPI()

config = Configuration()

init_config(config)


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
):
    try:
        load_from_local_files(
            paths_or_directory=paths,
            collection_name=collection_name,
            collection_description=collection_description,
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
):
    try:
        load_from_website(
            urls=urls,
            collection_name=collection_name,
            collection_description=collection_description,
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
        result = query(original_query, max_iter)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
