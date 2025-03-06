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

        existing_ids = collection.get(ids=[doc_id])["ids"]
        # if existing_ids:
        #     print(f"Skipping existing embedding ID: {doc_id}")
        #     continue

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
    print(" Database has been generated/updated.")

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

def refine_query(original_query, model="llama3.2"):
    reasoning_prompt = (
        "We have a short or ambiguous user query:\n"
        f"'{original_query}'\n\n"
        "Step-by-step, consider potential expansions, synonyms, or clarifications.\n"
        "Examples: if the query is 'info', possible expansions could be 'information', 'additional info', etc.\n"
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

if __name__ == "__main__":
    excel_path       = "clean.xlsx"
    vector_db_folder = "./vector_db"
    collection_name  = "TimesheetData"

    generateDB(excel_path, vector_db_folder, collection_name)

    user_query = "bacgoun"

    refined = refine_query(user_query, model="llama3.2")
    print(f" Original query: '{user_query}'")
    print(f" Refined query : '{refined}'")

    results = get_embeddings(refined, vector_db_folder, collection_name, top_n=3)

    print(f"\n Top matches for refined query: '{refined}'\n")
    for r in results:
        print("----")
        print(f" Doc ID: {r['doc_id']}")
        print(f" Timesheet Comment: {r['comment']}")
        print(f" Project Code: {r['prj_code']}")
        print(f" Project Name: {r['prj_name']}")
        print()

    combined_context = "\n".join([f"- {d['text']}" for d in results])
    final_answer = answer_with_llama(refined, combined_context, model="llama3.2")
    print(" LLM Answer:\n", final_answer)