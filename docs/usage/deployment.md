# 🌐 Deployment

This guide explains how to deploy DeepSearcher as a web service.

## ⚙️ Configure Modules

You can configure all arguments by modifying the configuration file:

```yaml
# config.yaml - https://github.com/zilliztech/deep-searcher/blob/main/config.yaml
llm:
  provider: "OpenAI"
  api_key: "your_openai_api_key_here"
  # Additional configuration options...
```

> **Important:** Set your `OPENAI_API_KEY` in the `llm` section of the YAML file.

## 🚀 Start Service

The main script will run a FastAPI service with default address `localhost:8000`:

```shell
$ python main.py
```

Once started, you should see output indicating the service is running successfully.

## 🔍 Access via Browser

You can access the web service through your browser:

1. Open your browser and navigate to [http://localhost:8000/docs](http://localhost:8000/docs)
2. The Swagger UI will display all available API endpoints
3. Click the "Try it out" button on any endpoint to interact with it
4. Fill in the required parameters and execute the request

This interactive documentation makes it easy to test and use all DeepSearcher API functionality. 