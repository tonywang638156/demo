import os
import json
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import ollama
import chromadb

# ----------------------------
# Configuration & Environment
# ----------------------------
load_dotenv()
DEFAULT_TIMESHEET_FILEPATH = os.getenv("TIMESHEET_FILEPATH", "./clean.xlsx")
CHROMADB_PATH = os.getenv("VECTOR_DATABASE_PATH", "./ts-cm-db6")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "TimesheetData")
CACHE_FILE = "expanded_cache.json"  # Cache file for LLM expansions

# ----------------------------
# Utility functions for caching
# ----------------------------
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as file:
            return json.load(file)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w") as file:
        json.dump(cache, file, indent=4)

# ----------------------------
# LLM-based comment enrichment
# ----------------------------
def expand_comment_with_llm(comment, cache):
    if comment in cache:
        return cache[comment]
    
    prompt = (
        f'The following is a short timesheet comment: "{comment}".\n'
        "Expand it into a detailed description (within 50 characters) explaining what this work might involve.\n"
        "Be professional, clear, and do not include any extra text.\n"
        "Only output the final expanded comment itself, nothing else."
    )
    
    response = ollama.generate(model="llama3.2", prompt=prompt)
    expanded_comment = response["response"]
    
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
    expanded_cache = load_cache()

    # Fetch existing IDs from ChromaDB
    existing_docs = collection.get(limit=None)
    existing_ids = set()
    for sublist in existing_docs["ids"]:
        existing_ids.update(sublist)

    for i, row in df.iterrows():
        doc_id = f"row-{i}"
        if doc_id in existing_ids:
            continue

        short_comment = str(row.get("trn_desc", "")).strip()
        expanded_comment = expand_comment_with_llm(short_comment, expanded_cache)
        st.write(f"Processing {doc_id}:")
        st.write(f"> Original: {short_comment}")
        st.write(f"> Expanded: {expanded_comment}")

        embed_response = ollama.embeddings(model="mxbai-embed-large", prompt=expanded_comment)
        embedding = embed_response["embedding"]

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
        st.write(f"Inserted new embedding ID: {doc_id}")

    st.success("Timesheet database updated.")

# ----------------------------
# Query functions: Refinement and Retrieval
# ----------------------------
def refine_query(original_query, model="llama3.2"):
    reasoning_prompt = (
        "We have a user query:\n"
        f"'{original_query}'\n\n"
        "Step-by-step, list possible clarifications or synonyms for key terms in the query.\n"
        "Then, produce ONE refined query (a single line) that makes the query clearer while preserving its original intent.\n"
        "Do not add extra context, greetings, or unrelated details—only output the refined query.\n"
    )
Do not add extra context, greetings, or unrelated details—only output the refined query.

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
    embed_response = ollama.embeddings(model="mxbai-embed-large", prompt=query)
    query_vector = embed_response["embedding"]

    db_response = collection.query(
        query_embeddings=[query_vector],
        n_results=top_n
    )
    
    matched_docs = []
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
# Streamlit App Interface
# ----------------------------
def main():
    st.title("Timesheet LLM & ChromaDB Demo")

    st.markdown("""
    This demo demonstrates a system that enriches timesheet comments with an LLM, embeds them, 
    stores them in a vector database (ChromaDB), and allows querying through refined queries.
    """)

    # Section for updating the Excel file
    st.header("Update Timesheet Data")
    st.markdown("Upload a new Excel file to update the timesheet data. This file will replace the default file used by the system.")
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])
    user_email = st.text_input("Enter your email address", help="Your email address is used to track who is updating the data. In a production system, this might be used for authentication or notifications.")

    if st.button("Update Database with Uploaded File"):
        if not uploaded_file:
            st.error("Please upload an Excel file first.")
        elif not user_email:
            st.error("Please enter your email address to proceed.")
        else:
            try:
                df = pd.read_excel(uploaded_file)
                # Optionally, you could save this file locally if needed:
                df.to_excel(DEFAULT_TIMESHEET_FILEPATH, index=False)
            except Exception as e:
                st.error(f"Error processing the Excel file: {e}")
                return

            collection = get_chroma_collection(CHROMADB_PATH, COLLECTION_NAME)
            with st.spinner("Processing timesheet data..."):
                generate_timesheet_db(df, collection)
            st.success("Database updated successfully with your uploaded file!")

    st.markdown("---")

    # Section for query processing
    st.header("Query the Timesheet Database")
    original_query = st.text_input("Enter your query (e.g., a vague term or code):", value="bacgoun")

    if st.button("Search") and original_query.strip() != "":
        collection = get_chroma_collection(CHROMADB_PATH, COLLECTION_NAME)
        with st.spinner("Refining query..."):
            refined_query = refine_query(original_query)
        st.write(f"**Refined Query:** {refined_query}")

        with st.spinner("Retrieving matching documents..."):
            results = get_embeddings(refined_query, collection, top_n=3)
        
        if results:
            st.subheader("Top Matching Documents")
            for r in results:
                st.markdown(f"**Doc ID:** {r['doc_id']}")
                st.markdown(f"- **Timesheet Comment:** {r['comment']}")
                st.markdown(f"- **Enriched Comment:** {r['enriched_comment']}")
                st.markdown(f"- **Project Code:** {r['prj_code']}")
                st.markdown(f"- **Project Name:** {r['prj_name']}")
                st.markdown("---")
        else:
            st.warning("No matching documents found.")

        # Combine context and get LLM answer
        combined_context = "\n".join([f"- {d['text']}" for d in results])
        with st.spinner("Generating answer..."):
            final_answer = answer_with_llama(refined_query, combined_context)
        st.subheader("LLM Answer")
        st.write(final_answer)

if __name__ == "__main__":
    main()
