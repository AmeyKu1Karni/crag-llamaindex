# Corrective-RAG with LlamaIndex, Pinecone, Tavily, and Hugging Face

This repository contains a Jupyter Notebook (`CRAG_website_SLM.ipynb`) demonstrating the implementation of a Corrective-Retrieval Augmented Generation (CRAG) pipeline using LlamaIndex.

The pipeline leverages:
*   **LlamaIndex:** As the core framework for building the RAG application and workflow.
*   **Pinecone:** As the vector database for storing and retrieving document embeddings.
*   **Hugging Face:**
    *   `BAAI/bge-base-en-v1.5` for generating text embeddings.
    *   `microsoft/Phi-3-mini-4k-instruct` (via Inference API) as the Large Language Model (LLM) for relevance evaluation, query transformation, and final answer generation.
*   **Tavily AI:** As a search tool to fetch up-to-date information or correct irrelevant retrievals from the vector store.

## Core Concept: Corrective-RAG (CRAG)

Traditional RAG retrieves documents based on initial query similarity and feeds them directly to the LLM. CRAG adds self-correction and enhancement steps:

1.  **Retrieve:** Initial documents are retrieved from a knowledge base (Pinecone vector store).
2.  **Evaluate Relevance:** The retrieved documents are assessed for their relevance to the query using an LLM.
3.  **Ambiguity Check & Query Transformation:** If documents are deemed irrelevant or insufficient, the original query is transformed into a more effective search engine query.
4.  **Web Search (Corrective Step):** The transformed query is used to search the web (using Tavily) for potentially more relevant or up-to-date information.
5.  **Generate:** The final answer is generated using the relevant documents from the initial retrieval *and* the results from the web search (if performed).

This approach aims to improve the robustness and accuracy of RAG systems by actively identifying and correcting potentially irrelevant retrievals.

## Technology Stack

*   **Framework:** LlamaIndex (`llama-index`)
*   **Vector Store:** Pinecone (`llama-index-vector-stores-pinecone`, `pinecone-client`)
*   **Embedding Model:** Hugging Face `BAAI/bge-base-en-v1.5` (`llama-index-embeddings-huggingface`, `sentence-transformers`)
*   **LLM:** Hugging Face `microsoft/Phi-3-mini-4k-instruct` (via Inference API) (`llama-index-llms-huggingface-api`)
*   **Web Search Tool:** Tavily AI (`llama-index-tools-tavily-research`, `tavily-python`)
*   **Core Libraries:** `torch`, `transformers`, `accelerate`, `bitsandbytes`
*   **Workflow:** LlamaIndex Workflow Engine
*   **Environment:** Developed in Google Colab (uses `nest_asyncio`)

## Features

*   **Document Ingestion:** Loads documents from a local `./data` directory.
*   **Vector Indexing:** Creates embeddings and stores them in a Pinecone serverless index (`crag`).
*   **Initial Retrieval:** Retrieves candidate documents from Pinecone based on the user query.
*   **LLM-based Relevance Evaluation:** Grades each retrieved document as "yes" or "no" for relevance.
*   **Conditional Query Transformation:** Rewrites the user query if irrelevant documents are detected.
*   **Conditional Web Search:** Uses Tavily Search API with the transformed query if correction is needed.
*   **Answer Synthesis:** Generates a final answer based on the filtered internal documents and external web search results.
*   **Asynchronous Workflow:** Uses LlamaIndex's asynchronous workflow engine for managing the steps.

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install llama-index llama-index-tools-tavily-research llama-index-embeddings-huggingface llama-index-llms-huggingface-api llama-index-vector-stores-pinecone transformers torch sentence-transformers pinecone-client tavily-python llama-index-readers-file accelerate bitsandbytes nest_asyncio
    ```
    *(Note: The notebook installs specific versions, you might want to create a `requirements.txt` file for exact replication).*

4.  **API Keys:** You will need API keys for:
    *   **Pinecone:** Get from [pinecone.io](https://www.pinecone.io/)
    *   **Tavily AI:** Get from [tavily.com](https://tavily.com/)
    *   **Hugging Face:** Get a User Access Token with at least `read` permissions from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

    The notebook uses `google.colab.userdata` which is specific to Colab. For local execution, you should set these as environment variables or use a `.env` file (and install `python-dotenv`):

    *   **Environment Variables:**
        ```bash
        export PINECONE_API_KEY="your_pinecone_api_key"
        export TAVILY_API_KEY="your_tavily_api_key"
        export HF_TOKEN="your_huggingface_token"
        ```
    *   **Modify the Code:** Alternatively, replace the `userdata.get(...)` calls in the notebook with direct key strings (less secure) or load them using `os.getenv` or `dotenv`.

5.  **Data:**
    *   Create a directory named `data` in the root of the project:
        ```bash
        mkdir data
        ```
    *   Place the documents you want to ingest into this `data/` directory. The `SimpleDirectoryReader` will load supported file types from here.

## Usage

1.  **Run the Jupyter Notebook:** Launch Jupyter Lab or Jupyter Notebook:
    ```bash
    jupyter lab
    # or
    jupyter notebook
    ```
    Open the `CRAG_website_SLM.ipynb` notebook.

2.  **Execute the Cells:** Run the cells sequentially.
    *   **Setup & Installation:** The initial cells install packages and import libraries.
    *   **API Keys:** Ensure your API keys are correctly configured (see Setup step 4).
    *   **Pinecone Initialization:** The notebook will connect to Pinecone and create the `crag` index if it doesn't exist.
    *   **Workflow Definition:** Defines the `CorrectiveRAGWorkflow`.
    *   **Ingestion/Index Loading:**
        *   If the Pinecone index is empty and the `data/` directory contains files, the `ingest` step will run, embedding and uploading your documents. This might take some time depending on the number of documents.
        *   If the index already contains vectors, it will load the existing index.
    *   **Querying:** The final cell demonstrates how to run a query through the workflow:
        ```python
        from IPython.display import Markdown, display

        response = await workflow.run(
            query_str="Your question here?", # Replace with your query
            index=index,
            tavily_ai_apikey=tavily_ai_apikey, # Ensure this variable holds your key
        )
        display(Markdown(str(response)))
        ```
        The output will show the relevance evaluation steps, any corrective actions (query transformation, Tavily search), and the final synthesized answer.

## Workflow Breakdown

The `CorrectiveRAGWorkflow` follows these steps:

1.  `ingest` (Optional): If documents are provided, embeds and indexes them into Pinecone.
2.  `prepare_for_retrieval`: Initializes the LLM, embedding model, API keys, and query/prompt templates.
3.  `retrieve`: Fetches initial candidate documents from the Pinecone index.
4.  `eval_relevance`: Uses the LLM to score the relevance of each retrieved document ("yes"/"no").
5.  `extract_relevant_texts`: Filters out documents marked as "no".
6.  `transform_query_pipeline`: If any document was marked "no", transforms the original query for web search and executes the search using Tavily.
7.  `query_result`: Synthesizes the final answer using a `SummaryIndex` over the relevant retrieved documents and any web search results.

## Customization

*   **LLM:** Change the `model_name` in the `HuggingFaceInferenceAPI` call to use a different model available via the API (consider size limits for the free tier). You could also switch to other LlamaIndex LLM integrations (e.g., `llama-index-llms-openai`).
*   **Embedding Model:** Modify the `HuggingFaceEmbedding` initialization. Ensure the chosen model's dimension matches the Pinecone index dimension (768 in this case).
*   **Vector Store:** Replace `PineconeVectorStore` with other supported vector stores.
*   **Prompts:** Adjust the `DEFAULT_RELEVANCY_PROMPT_TEMPLATE` and `DEFAULT_TRANSFORM_QUERY_TEMPLATE` for different LLM behaviors or specific needs.
*   **Retriever:** Modify retriever parameters (e.g., `similarity_top_k`) in the `index.as_retriever(**retriever_kwargs)` call.

## License

[Specify your license here, e.g., MIT License]