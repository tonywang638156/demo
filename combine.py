import os
import json
import pandas as pd
from dotenv import load_dotenv
import ollama
import chromadb

# --- Configuration ---
load_dotenv()
TIMESHEET_FILEPATH = os.getenv("TIMESHEET_FILEPATH", "./clean.xlsx")
CHROMADB_PATH = os.getenv("VECTOR_DATABASE_PATH", "./ts-cm-db6")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "TimesheetData")
CACHE_FILE = "expanded_cache.json"  # JSON file to store LLM-generated expansions

# ----------------------------
# Utility functions for caching
# ----------------------------
def load_cache():
    """Load cached expanded comments from a JSON file to avoid redundant LLM calls."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as file:
            return json.load(file)
    return {}

def save_cache(cache):
    """Save updated cache to the JSON file."""
    with open(CACHE_FILE, "w") as file:
        json.dump(cache, file, indent=4)

# ----------------------------
# LLM-based comment enrichment
# ----------------------------
def expand_comment_with_llm(comment, cache):
    """
    Enrich the short timesheet comment using an LLM.
    Uses caching to avoid redundant LLM calls.
    """
    if comment in cache:
        return cache[comment]
    
    # prompt = (
    #     f'The following is a short and vague timesheet comment: "{comment}".\n'
    #     "Expand it into a detailed description (within 100 characters) explaining what this work might involve.\n"
    #     "Be professional and clear. Provide the response in one paragraph, no lines in between"
    # )
    prompt = (
    f'The following is a short timesheet comment: "{comment}".\n'
    "Expand it into a detailed description (within 50 characters) explaining what this work might involve.\n"
    "Be professional, clear, and do not include any extra text.\n"
    "Only output the final expanded comment itself, nothing else."
    )
    
    response = ollama.generate(model="llama3.2", prompt=prompt)
    expanded_comment = response["response"]
    
    # Cache the result and save to file
    cache[comment] = expanded_comment
    save_cache(cache)
    
    return expanded_comment

# ----------------------------
# ChromaDB utility
# ----------------------------
def get_chroma_collection(db_path, collection_name):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)
    return collection

# ----------------------------
# Database generation: Enrich, embed, and store
# ----------------------------
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
            #print(f"Skipping existing embedding ID: {doc_id}")
            continue

        # Get original comment and expand using cache or LLM
        short_comment = str(row.get("trn_desc", "")).strip()
        expanded_comment = expand_comment_with_llm(short_comment, expanded_cache)
        print(("-----------------------------------------------------------------------"))
        print(f"orginal Comment for {doc_id}: {short_comment}")
        print(f"Expanded Comment for {doc_id}: {expanded_comment}")

        # Embed the expanded comment
        embed_response = ollama.embeddings(model="mxbai-embed-large", prompt=expanded_comment)
        embedding = embed_response["embedding"]

        # Store in ChromaDB
        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[short_comment],
            metadatas=[{
                "comment": short_comment,
                "enriched_comment": expanded_comment,
                "prj_code": str(row.get("prj_code", "")),
                "prj_name": str(row.get("prj_name", ""))
            }]
        )

        print(f"Inserted new embedding ID: {doc_id}")

    print("Timesheet database has been updated with cached LLM-enhanced descriptions.")

# ----------------------------
# Query functions: Refinement and Retrieval
# ----------------------------
def refine_query(original_query, model="llama3.2"):
    """
    Refine or translate a vague user query into a more precise one using an LLM.
    """
    reasoning_prompt = (
        "We have a short or ambiguous user query:\n"
        f"'{original_query}'\n\n"
        "Step-by-step, consider potential expansions, synonyms, or clarifications.\n"
        "For example, if the query is 'info', possible expansions could be 'information', 'additional info', etc.\n"
        "If the query is 'fin', expansions might be 'finance', 'financial data', etc.\n"
        "\n"
        "List these possibilities or your chain-of-thought reasoning, and end with a short summary.\n"
    )

    reasoning_response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": reasoning_prompt}],
    )
    reasoning_text = reasoning_response["message"]["content"]

    rewrite_prompt = (
        "Based on the following reasoning:\n\n"
        f"{reasoning_text}\n\n"
        "Now produce ONE refined query (a single line) that best captures the user's intent. "
        "Do not add disclaimers, greetings, or extra text. Only provide the final refined query.\n"
    )

    rewrite_response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": rewrite_prompt}],
    )
    final_refined = rewrite_response["message"]["content"].strip()
    return final_refined

def get_embeddings(query, collection, top_n=5):
    """
    Retrieve top matching documents for the given query from ChromaDB.
    """
    embed_response = ollama.embeddings(model="mxbai-embed-large", prompt=query)
    query_vector = embed_response["embedding"]

    db_response = collection.query(
        query_embeddings=[query_vector],
        n_results=top_n
    )
    
    matched_docs = []
    # Assume a single query so we iterate over the first sublist
    for i in range(len(db_response["documents"][0])):
        doc_text = db_response["documents"][0][i]
        doc_id = db_response["ids"][0][i]
        metadata = db_response["metadatas"][0][i]
        matched_docs.append({
            "doc_id": doc_id,
            "text": doc_text,
            "comment": metadata.get("comment", ""),
            "enriched_comment": metadata.get("enriched_comment", ""),
            "prj_code": metadata.get("prj_code", ""),
            "prj_name": metadata.get("prj_name", "")
        })
    
    return matched_docs

def answer_with_llama(user_query, context, model="llama3.2"):
    """
    Generate a final answer using an LLM based on the retrieved context.
    """
    rag_prompt = (
        f"You are given the following retrieved context from a timesheet database:\n\n"
        f"{context}\n\n"
        f"User query: {user_query}\n\n"
        "Please provide a detailed yet concise answer based on the provided context."
    )

    response = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': rag_prompt}]
    )
    return response['message']['content']

# ----------------------------
# Main execution
# ----------------------------
def main():
    # 1) Load timesheet data from Excel
    df = pd.read_excel(TIMESHEET_FILEPATH)

    # 2) Create or get ChromaDB collection
    collection = get_chroma_collection(CHROMADB_PATH, COLLECTION_NAME)

    # 3) Generate (or update) the timesheet database: only new rows will be processed
    generate_timesheet_db(df, collection)

    # 4) Process a user query
    original_query = "bacgoun"  # Example vague query
    refined_query = refine_query(original_query, model="llama3.2")
    print(f"Original query: '{original_query}'")
    print(f"Refined query : '{refined_query}'")

    # 5) Retrieve top matching documents from the DB
    results = get_embeddings(refined_query, collection, top_n=3)
    print(f"\nTop matches for refined query: '{refined_query}'\n")
    for r in results:
        print("----")
        print(f"Doc ID         : {r['doc_id']}")
        print(f"Timesheet Comment: {r['comment']}")
        print(f"Enriched Comment: {r['enriched_comment']}")
        print(f"Project Code   : {r['prj_code']}")
        print(f"Project Name   : {r['prj_name']}")
        print()

    # 6) Compose combined context from matched documents and get final answer
    combined_context = "\n".join([f"- {d['text']}" for d in results])
    final_answer = answer_with_llama(refined_query, combined_context, model="llama3.2")
    print("LLM Answer:\n", final_answer)

if __name__ == "__main__":
    main()
