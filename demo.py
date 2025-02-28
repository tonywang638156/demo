import os
import pandas as pd
import ollama
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
TIMESHEET_FILEPATH = os.getenv("TIMESHEET_FILEPATH")  # e.g. "clean.xlsx"
CHROMADB_PATH = os.getenv("VECTOR_DATABASE_PATH")      # e.g. "timesheet-chroma"
COLLECTION_NAME = "Timesheet-Comments"


def expand_comment_with_llm(comment):
    """
    Uses an LLM to generate a more detailed description of the short timesheet comment.
    """
    prompt = f"""
    The following is a short and vague timesheet comment: "{comment}".
    Expand it into a detailed description explaining what this work might involve.
    Be professional and clear.
    """
    
    response = ollama.generate(model="mistral", prompt=prompt)  # Change model if needed
    return response["response"]  # Extract LLM-generated text

def generate_timesheet_db(df, collection):
    # 1) Fetch all existing IDs in the collection
    existing_docs = collection.get()  # or collection.get(where={}) 
    existing_ids = []
    for sublist in existing_docs["ids"]:
        existing_ids.extend(sublist)
    existing_ids_set = set(existing_ids)

    # 2) Loop through each row in the DataFrame
    for i, row in df.iterrows():
        doc_id = f"row-{i}"

        # 3) If this doc_id is already present, skip re-embedding
        if doc_id in existing_ids_set:
            print(f"Skipping existing embedding ID: {doc_id}")
            continue

        # Otherwise embed and add
        comment = str(row.get("trn_desc", ""))
        prj_code = str(row.get("prj_code", ""))
        prj_name = str(row.get("prj_name", ""))

        embed_response = ollama.embeddings(model="mxbai-embed-large", prompt=comment)
        embedding = embed_response["embedding"]

        doc_text = f"Timesheet Comment: {comment}"
        metadata = {
            "prj_code": prj_code,
            "prj_name": prj_name,
        }

        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[doc_text],
            metadatas=[metadata]
        )
        print(f"Inserted new embedding ID: {doc_id}")

    print("Timesheet database has been updated (only new rows were added).")


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
