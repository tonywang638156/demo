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

def get_chroma_collection(db_path, collection_name):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)
    return collection

def generateDB(filepath, db_path, collection_name):
    timesheet_comments, project_codes, project_names = load_timesheet_data(filepath)
    collection = get_chroma_collection(db_path, collection_name)

    for i, comment in enumerate(timesheet_comments):
        doc_id = f"row_{i}"
        doc_text = (
            f"Timesheet Comment: {comment}\n"
            f"Project Code: {project_codes[i]}\n"
            f"Project Name: {project_names[i]}"
        )
        
        embedding_response = ollama.embeddings(model="mxbai-embed-large", prompt=comment)
        embedding_vector = embedding_response["embedding"]
        
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

def get_embeddings(prompt, db_path, collection_name, top_n=5):
    collection = get_chroma_collection(db_path, collection_name)
    embed_response = ollama.embeddings(model="mxbai-embed-large", prompt=prompt)
    query_vector = embed_response["embedding"]

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

def answer_with_llama(user_query, context, model="llama3.2"):
    rag_prompt = (
        f"You are given the following context from a timesheet database:\n\n"
        f"{context}\n\n"
        f"User query: {user_query}\n\n"
        f"Please provide an appropriate answer or summary."
    )
    response = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': rag_prompt}]
    )
    return response['message']['content']


# -----------------------------------------------------
# NEW FUNCTION: QUERY REFINEMENT (QUERY FUSION / MULTI-QUERY)
# -----------------------------------------------------
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


if __name__ == "__main__":
    excel_path       = "dbo_Prj_Detail_Charges.xlsx"
    vector_db_folder = "./vector_db"
    collection_name  = "TimesheetData"

    # Build or update the DB
    generateDB(excel_path, vector_db_folder, collection_name)

    # Original user query: ambiguous or short
    user_query = "background"

    # 1) REFINE the user query with the LLM
    refined = refine_query(user_query, model="llama3.2")
    print(f"Original query: '{user_query}'")
    print(f"Refined query : '{refined}'")

    # 2) SEARCH with the refined query
    results = get_embeddings(refined, vector_db_folder, collection_name, top_n=3)

    print(f"\nTop matches for refined query: '{refined}'\n")
    for r in results:
        print("----")
        print(f"Doc ID: {r['doc_id']}")
        print(f"Timesheet Comment: {r['comment']}")
        print(f"Project Code     : {r['prj_code']}")
        print(f"Project Name     : {r['prj_name']}")
        print()

    # 3) Combine context and get final LLM-based answer
    combined_context = "\n".join([f"- {d['text']}" for d in results])
    final_answer = answer_with_llama(refined, combined_context, model="llama3.2")
    print("LLM Answer:\n", final_answer)
