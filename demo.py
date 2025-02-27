import os
import sys
import pandas as pd
import ollama
import chromadb


def load_timesheet_data(filepath):
    df = pd.read_excel(filepath)
    timesheet_comments = df["trn_desc"].tolist()
    project_codes      = df["prj_code"].tolist()
    project_names      = df["prj_name"].tolist()
    return timesheet_comments, project_codes, project_names

# ---------------------------------------------------------------------
# 2) CREATE/OPEN CHROMA CLIENT AND COLLECTION
# ---------------------------------------------------------------------
def get_chroma_collection(db_path, collection_name):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)
    return collection

# ---------------------------------------------------------------------
# 3) GENERATE DB: EMBED TIMESHEET ROWS AND STORE IN CHROMADB IF NEEDED
def generateDB(filepath, db_path, collection_name):
    timesheet_comments, project_codes, project_names = load_timesheet_data(filepath)
    collection = get_chroma_collection(db_path, collection_name)

    print("🛠 Checking existing embeddings in ChromaDB...")

    for i, comment in enumerate(timesheet_comments):
        doc_id = f"row_{i}"  # Unique ID for each document

        # Check if this specific document ID already exists in ChromaDB
        existing_ids = collection.get(ids=[doc_id])["ids"]
        if existing_ids:
            print(f"✅ Skipping already stored embedding: {doc_id}")
            continue  # Skip embedding if it already exists

        # Generate embedding only if it's a new entry
        embedding_response = ollama.embeddings(model="mxbai-embed-large", prompt=comment)
        embedding_vector = embedding_response["embedding"]

        # Store in ChromaDB
        collection.add(
            ids=[doc_id],
            embeddings=[embedding_vector],
            documents=[comment],
            metadatas=[{
                "comment": comment,
                "prj_code": project_codes[i],
                "prj_name": project_names[i]
            }]
        )
        print(f"📌 Inserted new embedding ID: {doc_id}")

    print("✅ Database updated with only new embeddings.")


# ---------------------------------------------------------------------
# 4) QUERY FUNCTION: RETRIEVE EMBEDDINGS FROM CHROMA
# ---------------------------------------------------------------------
def get_embeddings(prompt, db_path, collection_name, top_n=5):
    collection = get_chroma_collection(db_path, collection_name)
    
    # Generate embedding for user query
    embed_response = ollama.embeddings(model="mxbai-embed-large", prompt=prompt)
    query_vector = embed_response["embedding"]

    # Query ChromaDB for relevant results
    dbResponse = collection.query(
        query_embeddings=[query_vector],
        n_results=top_n
    )

    matched_docs = []
    for i in range(len(dbResponse["documents"][0])):
        doc_text   = dbResponse["documents"][0][i]
        doc_id     = dbResponse["ids"][0][i]
        metadata   = dbResponse["metadatas"][0][i]
        matched_docs.append({
            "doc_id": doc_id,
            "text": doc_text,
            "comment": metadata["comment"],
            "prj_code": metadata["prj_code"],
            "prj_name": metadata["prj_name"]
        })
    
    return matched_docs

# ---------------------------------------------------------------------
# 5) QUERY REFINEMENT: HANDLE SHORT OR VAGUE QUERIES
# ---------------------------------------------------------------------
def refine_query(original_query, model="llama3.2"):
    """
    Uses a multi-step reasoning-based approach to rewrite short or ambiguous queries.
    """
    reasoning_prompt = (
        "We have a short or ambiguous user query:\n"
        f"'{original_query}'\n\n"
        "Step-by-step, consider potential expansions, synonyms, or clarifications.\n"
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

# ---------------------------------------------------------------------
# 6) GENERATE FINAL LLM RESPONSE
# ---------------------------------------------------------------------
def answer_with_llama(user_query, context, model="llama3.2"):
    rag_prompt = (
        f"You are given the following retrieved context from a timesheet database:\n\n"
        f"{context}\n\n"
        f"User query: {user_query}\n\n"
        f"Please provide a detailed yet concise answer based on the provided context."
    )

    response = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': rag_prompt}]
    )
    return response['message']['content']

# ---------------------------------------------------------------------
# 7) MAIN EXECUTION: RUN SYSTEM
# ---------------------------------------------------------------------
if __name__ == "__main__":
    excel_path       = "dbo_Prj_Detail_Charges.xlsx"
    vector_db_folder = "./vector_db"
    collection_name  = "TimesheetData"

    # Generate or update the database (only if needed)
    generateDB(excel_path, vector_db_folder, collection_name)

    # 1) Take the raw user query
    user_query = "bg"  # Example of a vague or shorthand query

    # 2) Refine the query using LLM
    refined = refine_query(user_query, model="llama3.2")
    print(f"🔍 Original query: '{user_query}'")
    print(f"✅ Refined query : '{refined}'")

    # 3) Use refined query for embedding-based search
    results = get_embeddings(refined, vector_db_folder, collection_name, top_n=3)

    print(f"\n🔎 Top matches for refined query: '{refined}'\n")
    for r in results:
        print("----")
        print(f"📌 Doc ID: {r['doc_id']}")
        print(f"📝 Timesheet Comment: {r['comment']}")
        print(f"📁 Project Code: {r['prj_code']}")
        print(f"📌 Project Name: {r['prj_name']}")
        print()

    # 4) Generate final LLM response
    combined_context = "\n".join([f"- {d['text']}" for d in results])
    final_answer = answer_with_llama(refined, combined_context, model="llama3.2")
    print("💡 LLM Answer:\n", final_answer)
