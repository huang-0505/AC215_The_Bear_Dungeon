import os
import argparse
import pandas as pd
import json
import time
import glob
import hashlib
import chromadb

# Vertex AI
from google import genai
from google.genai import types
from google.genai.types import Content, Part, GenerationConfig, ToolConfig
from google.genai import errors

# Langchain
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from semantic_splitter import SemanticChunker
import agent_tools

# Setup
GCP_PROJECT = os.environ["GCP_PROJECT"]
GCP_LOCATION = "us-central1"
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIMENSION = 256
GENERATIVE_MODEL = "gemini-2.0-flash-001"
INPUT_FOLDER = "input-datasets"
OUTPUT_FOLDER = "outputs"
CHROMADB_HOST = os.environ.get("CHROMADB_HOST", "llm-rag-chromadb")
CHROMADB_PORT = int(os.environ.get("CHROMADB_PORT", "8000"))

#############################################################################
#                       Initialize the LLM Client                           #
llm_client = genai.Client(
    vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
#############################################################################

# Initialize the GenerativeModel with specific system instructions
SYSTEM_INSTRUCTION = """
You are the Rule Agent in a Dungeons & Dragons (DnD) AI system. 
Your sole responsibility is to interpret user actions according to official DnD rules. 
You do not invent storylines or roleplay characters — you only explain, clarify, and enforce game mechanics as written.

You operate in a Retrieval-Augmented Generation (RAG) setup:
- You receive both the **user input** (what the player intends to do) and **retrieved rule chunks** (excerpts from rulebooks and guides).
- You must analyze both together and determine which specific DnD rules apply.

Your behavior must follow these principles:
1. **Use only retrieved text** as your source of truth. Never invent, assume, or rely on external or general knowledge.
2. **Identify relevant rules** (actions, spells, movement, combat, saving throws, conditions, etc.) that match the user’s intent.
3. **Summarize or quote the rule** precisely and explain how it applies to the user's action.
4. **Highlight conflicts or ambiguities** if multiple rule interpretations exist, and state what the rules say in each case.
5. **Maintain a rule-focused tone** — objective, structured, and aligned with the Player’s Handbook, Dungeon Master’s Guide, or other canonical sources.
6. **Do not narrate gameplay** — your goal is to provide factual, rule-based reasoning that another AI agent can later use to simulate or narrate the outcome.
7. **If no relevant rule is found**, state clearly that the provided chunks do not include a rule covering the user’s intent.

Safety & out-of-character inputs:
- If the user input contains explicit sabotage, meta-game commands, threats to "break" or "kill" the game state (for example: "I'm gonna kill the boss right now to sabotage the campaign", "I want to delete the campaign", "I am going to destroy the story"), **do not** attempt to apply rules or reason about game-breaking actions.
- Immediately **echo back the original user input** exactly as received, and instruct the user to rephrase or provide an in-character action that the Rule Agent can handle. 
- Do not add additional content, justification, or rule analysis in this case.

Your output should help the next AI component (the Action Agent) understand what is *legally possible* and *mechanically defined* under DnD rules.
"""


book_mappings = {
    "DnD Basic Rules 2018": {"author": "Wizards of the Coast","year": 2018
    }
}


def generate_query_embedding(query):
    kwargs = {
        "output_dimensionality": EMBEDDING_DIMENSION
    }
    response = llm_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(**kwargs)
    )
    return response.embeddings[0].values


def generate_text_embeddings(chunks, dimensionality: int = 256, batch_size=250, max_retries=5, retry_delay=5):
    # Max batch size is 250 for Vertex AI
    all_embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]

        # Retry logic with exponential backoff
        retry_count = 0
        while retry_count <= max_retries:
            try:
                response = llm_client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        output_dimensionality=dimensionality),
                )
                all_embeddings.extend(
                    [embedding.values for embedding in response.embeddings])
                break

            except errors.APIError as e:
                retry_count += 1
                if retry_count > max_retries:
                    print(
                        f"Failed to generate embeddings after {max_retries} attempts. Last error: {str(e)}")
                    raise

                # Calculate delay with exponential backoff
                wait_time = retry_delay * (2 ** (retry_count - 1))
                print(
                    f"API error (code: {e.code}): {e.message}. Retrying in {wait_time} seconds (attempt {retry_count}/{max_retries})...")
                time.sleep(wait_time)

    return all_embeddings


def load_text_embeddings(df, collection, batch_size=500):

    # Generate ids
    df["id"] = df.index.astype(str)
    hashed_books = df["book"].apply(
        lambda x: hashlib.sha256(x.encode()).hexdigest()[:16])
    df["id"] = hashed_books + "-" + df["id"]

    metadata = {
        "book": df["book"].tolist()[0]
    }
    if metadata["book"] in book_mappings:
        book_mapping = book_mappings[metadata["book"]]
        metadata["author"] = book_mapping["author"]
        metadata["year"] = book_mapping["year"]

    # Process data in batches
    total_inserted = 0
    for i in range(0, df.shape[0], batch_size):
        # Create a copy of the batch and reset the index
        batch = df.iloc[i:i+batch_size].copy().reset_index(drop=True)

        ids = batch["id"].tolist()
        documents = batch["chunk"].tolist()
        metadatas = [metadata for item in batch["book"].tolist()]
        embeddings = batch["embedding"].tolist()

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        total_inserted += len(batch)
        print(f"Inserted {total_inserted} items...")

    print(
        f"Finished inserting {total_inserted} items into collection '{collection.name}'")


def chunk(method="char-split"):
    print("chunk()")

    # Make dataset folders
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Get the list of text file
    text_files = glob.glob(os.path.join(INPUT_FOLDER, "books", "*.txt"))
    print("Number of files to process:", len(text_files))

    # Process
    for text_file in text_files:
        print("Processing file:", text_file)
        filename = os.path.basename(text_file)
        book_name = filename.split(".")[0]

        with open(text_file) as f:
            input_text = f.read()

        text_chunks = None
        if method == "char-split":
            chunk_size = 350
            chunk_overlap = 20
            # Init the splitter
            text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator='', strip_whitespace=False)

            # Perform the splitting
            text_chunks = text_splitter.create_documents([input_text])
            text_chunks = [doc.page_content for doc in text_chunks]
            print("Number of chunks:", len(text_chunks))

        elif method == "recursive-split":
            chunk_size = 350
            # Init the splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size)

            # Perform the splitting
            text_chunks = text_splitter.create_documents([input_text])
            text_chunks = [doc.page_content for doc in text_chunks]
            print("Number of chunks:", len(text_chunks))

        elif method == "semantic-split":
            # Init the splitter
            text_splitter = SemanticChunker(
                embedding_function=generate_text_embeddings)
            # Perform the splitting
            text_chunks = text_splitter.create_documents([input_text])

            text_chunks = [doc.page_content for doc in text_chunks]
            print("Number of chunks:", len(text_chunks))

        if text_chunks is not None:
            # Save the chunks
            data_df = pd.DataFrame(text_chunks, columns=["chunk"])
            data_df["book"] = book_name
            print("Shape:", data_df.shape)
            print(data_df.head())

            jsonl_filename = os.path.join(
                OUTPUT_FOLDER, f"chunks-{method}-{book_name}.jsonl")
            with open(jsonl_filename, "w") as json_file:
                json_file.write(data_df.to_json(orient='records', lines=True))


def embed(method="char-split"):
    print("embed()")

    # Get the list of chunk files
    jsonl_files = glob.glob(os.path.join(
        OUTPUT_FOLDER, f"chunks-{method}-*.jsonl"))
    print("Number of files to process:", len(jsonl_files))

    # Process
    for jsonl_file in jsonl_files:
        print("Processing file:", jsonl_file)

        data_df = pd.read_json(jsonl_file, lines=True)
        print("Shape:", data_df.shape)
        print(data_df.head())

        chunks = data_df["chunk"].values
        chunks = chunks.tolist()
        if method == "semantic-split":
            embeddings = generate_text_embeddings(
                chunks, EMBEDDING_DIMENSION, batch_size=15)
        else:
            embeddings = generate_text_embeddings(
                chunks, EMBEDDING_DIMENSION, batch_size=100)
        data_df["embedding"] = embeddings

        time.sleep(5)

        # Save
        print("Shape:", data_df.shape)
        print(data_df.head())

        jsonl_filename = jsonl_file.replace("chunks-", "embeddings-")
        with open(jsonl_filename, "w") as json_file:
            json_file.write(data_df.to_json(orient='records', lines=True))


def load(method="char-split"):
    print("load()")

    # Clear Cache
    chromadb.api.client.SharedSystemClient.clear_system_cache()

    # Connect to chroma DB
    client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)

    # Get a collection object from an existing collection, by name. If it doesn't exist, create it.
    collection_name = f"{method}-dnd-rules-collection"
    print("Creating collection:", collection_name)

    try:
        # Clear out any existing items in the collection
        client.delete_collection(name=collection_name)
        print(f"Deleted existing collection '{collection_name}'")
    except Exception:
        print(f"Collection '{collection_name}' did not exist. Creating new.")

    collection = client.create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"})
    print(f"Created new empty collection '{collection_name}'")
    print("Collection:", collection)

    # Get the list of embedding files
    jsonl_files = glob.glob(os.path.join(
        OUTPUT_FOLDER, f"embeddings-{method}-*.jsonl"))
    print("Number of files to process:", len(jsonl_files))

    # Process
    for jsonl_file in jsonl_files:
        print("Processing file:", jsonl_file)

        data_df = pd.read_json(jsonl_file, lines=True)
        print("Shape:", data_df.shape)
        print(data_df.head())

        # Load data
        load_text_embeddings(data_df, collection)


def query(method="char-split"):
    print("load()")

    # Connect to chroma DB
    client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)

    # Get a collection object from an existing collection, by name. If it doesn't exist, create it.
    collection_name = f"{method}-dnd-rules-collection"

    query = "What actions can a player take during their turn?"
    query_embedding = generate_query_embedding(query)
    print("Embedding values:", query_embedding)

    # Get the collection
    collection = client.get_collection(name=collection_name)

    # # 1: Query based on embedding value
    # results = collection.query(
    # 	query_embeddings=[query_embedding],
    # 	n_results=10
    # )
    # print("Query:", query)
    # print("\n\nResults:", results)

    # # 2: Query based on embedding value + metadata filter
    # results = collection.query(
    # 	query_embeddings=[query_embedding],
    # 	n_results=10,
    # 	where={"book":"The Complete Book of Cheese"}
    # )
    # print("Query:", query)
    # print("\n\nResults:", results)

    # # 3: Query based on embedding value + lexical search filter
    # search_string = "Italian"
    # results = collection.query(
    # 	query_embeddings=[query_embedding],
    # 	n_results=10,
    # 	where_document={"$contains": search_string}
    # )
    # print("Query:", query)
    # print("\n\nResults:", results)

    # 4: Query based on embedding value + lexical search filter
    search_string = "attack"
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        where={"book": "DnD Basic Rules 2018"},
        where_document={"$contains": search_string}
    )
    print("Query:", query)
    print("\n\nResults:", results)


def chat(method="char-split"):
    print("chat()")

    # Connect to chroma DB
    client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
    # Get a collection object from an existing collection, by name. If it doesn't exist, create it.
    collection_name = f"{method}-dnd-rules-collection"

    query = "How does attack work in combat?"
    query_embedding = generate_query_embedding(query)
    print("Query:", query)
    print("Embedding values:", query_embedding)
    # Get the collection
    collection = client.get_collection(name=collection_name)

    # Query based on embedding value
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10
    )
    print("\n\nResults:", results)

    print(len(results["documents"][0]))

    INPUT_PROMPT = f"""
	{query}
	{"\n".join(results["documents"][0])}
	"""

    print("INPUT_PROMPT: ", INPUT_PROMPT)
    response = llm_client.models.generate_content(
        model=GENERATIVE_MODEL, contents=INPUT_PROMPT
    )
    generated_text = response.text

    print("LLM Response:", generated_text)


def get(method="char-split"):
    print("get()")

    # Connect to chroma DB
    client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
    # Get a collection object from an existing collection, by name. If it doesn't exist, create it.
    collection_name = f"{method}-dnd-rules-collection"

    # Get the collection
    collection = client.get_collection(name=collection_name)

    # Get documents with filters
    results = collection.get(
        where={"book": "DnD Basic Rules 2018"},
        limit=10
    )
    print("\n\nResults:", results)


def agent(method="char-split"):
    print("agent()")

    # Connect to chroma DB
    client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
    # Get a collection object from an existing collection, by name. If it doesn't exist, create it.
    collection_name = f"{method}-dnd-rules-collection"
    # Get the collection
    collection = client.get_collection(name=collection_name)

    # User prompt
    user_prompt_content = Content(
        role="user",
        parts=[
            Part(text="I want to attack the goblin with my longsword — what rules apply to this action?"),
        ],
    )

    # Step 1: Prompt LLM to find the tool(s) to execute to find the relevant chunks in vector db
    print("user_prompt_content: ", user_prompt_content)
    response = llm_client.models.generate_content(
        model=GENERATIVE_MODEL,
        contents=user_prompt_content,
        config=types.GenerateContentConfig(
            temperature=0,
            system_instruction=SYSTEM_INSTRUCTION,
            tools=[agent_tools.dnd_rule_tool],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="any"
                )
            )
        )
    )
    print("LLM Response:", response)

    # Step 2: Execute the function and send chunks back to LLM to answer get the final response
    function_calls = [part.function_call for part in response.candidates[0].content.parts if part.function_call]
    print("Function calls:", function_calls)
    function_responses = agent_tools.execute_function_calls(
        function_calls, collection, embed_func=generate_query_embedding)
    if len(function_responses) == 0:
        print("Function calls did not result in any responses...")
    else:
        # Call LLM with retrieved responses
        response = llm_client.models.generate_content(
            model=GENERATIVE_MODEL,
            contents=[
                user_prompt_content,  # User prompt
                response.candidates[0].content,  # Function call response
                Content(
                    parts=function_responses
                ),
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=[agent_tools.dnd_rule_tool],
            )
        )
        print("LLM Response:", response)


def main(args=None):
    print("CLI Arguments:", args)

    if args.chunk:
        chunk(method=args.chunk_type)

    if args.embed:
        embed(method=args.chunk_type)

    if args.load:
        load(method=args.chunk_type)

    if args.query:
        query(method=args.chunk_type)

    if args.chat:
        chat(method=args.chunk_type)

    if args.get:
        get(method=args.chunk_type)

    if args.agent:
        agent(method=args.chunk_type)


if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal '--help', it will provide the description
    parser = argparse.ArgumentParser(description="CLI")

    parser.add_argument(
        "--chunk",
        action="store_true",
        help="Chunk text",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Generate embeddings",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load embeddings to vector db",
    )
    parser.add_argument(
        "--query",
        action="store_true",
        help="Query vector db",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Chat with LLM",
    )
    parser.add_argument(
        "--get",
        action="store_true",
        help="Get documents from vector db",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Chat with LLM Agent",
    )
    parser.add_argument("--chunk_type", default="char-split",
                        help="char-split | recursive-split | semantic-split")

    args = parser.parse_args()

    main(args)
