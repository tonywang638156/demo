import os
import json
import pandas as pd
import ollama
import chromadb
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
TIMESHEET_FILEPATH = os.getenv("TIMESHEET_FILEPATH")  
CHROMADB_PATH = os.getenv("VECTOR_DATABASE_PATH")     
COLLECTION_NAME = "Timesheet-Comments"
CACHE_FILE = "expanded_cache.json"  # JSON file to store LLM-generated expansions


def load_cache():
    """Load cached expanded comments from a JSON file to avoid redundant LLM calls."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as file:
            return json.load(file)
    return {}  # Return an empty dictionary if cache file doesn't exist


def save_cache(cache):
    """Save updated cache to the JSON file."""
    with open(CACHE_FILE, "w") as file:
        json.dump(cache, file, indent=4)


def expand_comment_with_llm(comment, cache):
    """
    Expands the short timesheet comment using an LLM.
    Uses caching to avoid redundant LLM calls.
    """
    if comment in cache:
        return cache[comment]  # Reuse cached response

    prompt = f"""
    The following is a short and vague timesheet comment: "{comment}".
    Expand it into a detailed description explaining what this work might involve.
    Be professional and clear.
    """
    
    response = ollama.generate(model="mistral", prompt=prompt)  
    expanded_comment = response["response"]
    
    # Cache the result
    cache[comment] = expanded_comment
    save_cache(cache)

    return expanded_comment


def generate_timesheet_db(df, collection):
    """
    Enhances timesheet comments using LLM, embeds them, and stores them in ChromaDB.
    Uses a cache to speed up processing.
    """

    # Load cache
    expanded_cache = load_cache()

    # Fetch all existing IDs in ChromaDB
    existing_docs = collection.get(limit=None)
    existing_ids = []
    for sublist in existing_docs["ids"]:
        existing_ids.append(sublist)

    existing_ids_set = set(existing_ids)

    # Process each row
    for i, row in df.iterrows():
        doc_id = f"row-{i}"

        if doc_id in existing_ids_set:
            print(f"Skipping existing embedding ID: {doc_id}")
            continue

        # Get original comment and expand using cache or LLM
        short_comment = str(row.get("trn_desc", "")).strip()
        expanded_comment = expand_comment_with_llm(short_comment, expanded_cache)

        print(f"Expanded Comment for {doc_id}: {expanded_comment}")  

        # Embed the expanded comment
        embed_response = ollama.embeddings(model="mxbai-embed-large", prompt=expanded_comment)
        embedding = embed_response["embedding"]

        # Store in ChromaDB
        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[expanded_comment],
            metadatas=[{"prj_name": str(row.get("prj_name", ""))}]
        )

        print(f"Inserted new embedding ID: {doc_id}")

    print("Timesheet database has been updated with cached LLM-enhanced descriptions.")



def query_timesheet_db(query, collection, top_n=10):
    """
    Given a text query, embed it and retrieve the top_n
    matching documents from the DB. Then return a list
    of (prj_code, prj_name, timesheet_comment).
    """
    # 1) Embed the query
    query_embed = ollama.embeddings(model="mxbai-embed-large", prompt=query)
    query_embedding = query_embed["embedding"]

    # 2) Query the ChromaDB collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n
    )

    # 'results' is a dict with keys: "ids", "embeddings", "metadatas", "documents"
    # Each is a list of lists, e.g. results["metadatas"] -> [[{...}, {...}], [{...}, {...}]]
    # We'll flatten them for convenience:
    matched_projects = []
    for meta_list, doc_list in zip(results["metadatas"], results["documents"]):
        for meta, doc in zip(meta_list, doc_list):
            matched_projects.append({
                "prj_code": meta.get("prj_code"),
                "prj_name": meta.get("prj_name"),
                "comment": doc
            })

    return matched_projects

def main():
    # 1) Load the Timesheet data
    df = pd.read_excel(TIMESHEET_FILEPATH)

    # 2) Create/Get ChromaDB client + collection
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # (Optional) If you want a fresh rebuild every run, you can do:
    #   client.delete_collection(name=COLLECTION_NAME)
    #   collection = client.create_collection(name=COLLECTION_NAME)

    # 3) Generate (i.e. vectorize + store) the data
    generate_timesheet_db(df, collection)

    # 4) Test a query
    test_query = "Steel design"  # or "interior design"
    results = query_timesheet_db(test_query, collection, top_n=20)
    
    print(f"\nTop matches for query: '{test_query}'")
    for item in results:
        print(f"Project Code: {item['prj_code']}, Project Name: {item['prj_name']}")
        # If desired, also show timesheet comment:
        # print(f"  Comment: {item['comment']}\n")

if __name__ == "__main__":
    main()
