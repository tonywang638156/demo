import os
import sys
import pandas as pd
import ollama
import chromadb

# ---------------------------------------------------------------------
# 1) READ EXCEL DATA
# ---------------------------------------------------------------------
def load_timesheet_data(filepath):
    df = pd.read_excel(filepath)
    # Extract relevant columns
    timesheet_comments = df["trn_desc"].tolist()
    project_codes      = df["prj_code"].tolist()
    project_names      = df["prj_name"].tolist()
    return timesheet_comments, project_codes, project_names

# ---------------------------------------------------------------------
# 2) CREATE/OPEN CHROMA CLIENT AND COLLECTION
# ---------------------------------------------------------------------
def get_chroma_collection(db_path, collection_name):
    # Initialize Chroma client, pointing to a local folder for persistent storage
    client = chromadb.PersistentClient(path=db_path)
    # If the collection doesn’t exist, it will be created
    collection = client.get_or_create_collection(name=collection_name)
    return collection

# ---------------------------------------------------------------------
# 3) GENERATE DB: EMBED TIMESHEET ROWS AND STORE IN CHROMADB
# ---------------------------------------------------------------------
def generateDB(filepath, db_path, collection_name):
    # Load the timesheet data
    timesheet_comments, project_codes, project_names = load_timesheet_data(filepath)
    
    collection = get_chroma_collection(db_path, collection_name)

    # For each row, create an embedding via Ollama and add to Chroma
    for i, comment in enumerate(timesheet_comments):
        doc_id = f"row_{i}"  # unique ID for Chroma
        # Combine project info + comment if you want them all in one "document"
        doc_text = (
            f"Timesheet Comment: {comment}\n"
            f"Project Code: {project_codes[i]}\n"
            f"Project Name: {project_names[i]}"
        )
        
        # 3A) Get the embedding from Ollama
        #     We pass the timesheet comment as the "prompt" to the embedding endpoint:
        embedding_response = ollama.embeddings(model="mxbai-embed-large", prompt=comment)
        # The returned JSON typically has an "embedding" field
        embedding_vector = embedding_response["embedding"]

        # 3B) Add this to the Chroma collection
        collection.add(
            ids=[doc_id],
            embeddings=[embedding_vector],
            documents=[doc_text],
            metadatas=[{
                "comment": comment,
                "prj_code": project_codes[i],
                "prj_name": project_names[i]
            }]
        )
    print("Database has been generated/updated.")

# ---------------------------------------------------------------------
# 4) QUERY FUNCTION: GET EMBEDDINGS + SEARCH CHROMA
# ---------------------------------------------------------------------
def get_embeddings(prompt, db_path, collection_name, top_n=5):
    """
    1) Convert 'prompt' into an embedding using mxbai-embed-large
    2) Query the Chroma DB
    3) Return the top N results
    """
    collection = get_chroma_collection(db_path, collection_name)

    # Embed the query prompt
    embed_response = ollama.embeddings(model="mxbai-embed-large", prompt=prompt)
    query_vector = embed_response["embedding"]

    # Query top N relevant docs
    dbResponse = collection.query(
        query_embeddings=[query_vector],
        n_results=top_n
    )
    
    matched_docs = []
    for i in range(len(dbResponse["documents"][0])):  # how many results
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
# 5) OPTIONAL: Pass results to an LLM (like llama3.2)
# ---------------------------------------------------------------------
def refine_query(original_query, model="llama3.2"):
    """
    Use the LLM to rewrite or refine an ambiguous query (e.g. "bg") into a more specific one.
    """
    refinement_prompt = (
        f"You are a helpful AI that refines vague or shorthand queries.\n"
        f"The user query is: '{original_query}'\n\n"
        f"1. Interpret common abbreviations.\n"
        f"2. Rewrite the query to be more descriptive, assuming the user wants 'background' context.\n"
        f"3. If the query is already clear, leave it as is.\n\n"
        f"Now produce a more specific query for timesheet background information."
    )

    response = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': refinement_prompt}]
    )
    refined_query = response['message']['content'].strip()
    return refined_query


# ---------------------------------------------------------------------
# 6) MAIN DEMO
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Provide the file paths and collection name
    excel_path       = "dbo_Prj_Detail_Charges.xlsx"
    vector_db_folder = "./vector_db"  # folder for Chroma to store data
    collection_name  = "TimesheetData"

    # 6A) Generate the DB (run once or anytime data changes)
    generateDB(excel_path, vector_db_folder, collection_name)

    # 6B) Example query
    user_query = "steel design"
    results = get_embeddings(user_query, vector_db_folder, collection_name, top_n=5)

    print(f"\nTop matches for query: '{user_query}'\n")
    for r in results:
        print("----")
        print(f"Doc ID: {r['doc_id']}")
        print(f"Timesheet Comment: {r['comment']}")
        print(f"Project Code     : {r['prj_code']}")
        print(f"Project Name     : {r['prj_name']}")
        print()

    # 6C) If you want a final LLM-based answer:
    combined_context = "\n".join([f"- {d['text']}" for d in results])
    final_answer = answer_with_llama(user_query, combined_context, model="llama3.2:latest")
    print("LLM Answer:\n", final_answer)
