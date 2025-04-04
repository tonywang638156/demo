import os
import json
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import ollama
import chromadb

# ----------------------------
# Environment and file paths
# ----------------------------
load_dotenv()
TIMESHEET_FILEPATH = os.getenv("TIMESHEET_FILEPATH", "./clean.xlsx")
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
            continue

        # Get original comment and expand using cache or LLM
        short_comment = str(row.get("trn_desc", "")).strip()
        expanded_comment = expand_comment_with_llm(short_comment, expanded_cache)
        print("-----------------------------------------------------------------------")
        print(f"Original Comment for {doc_id}: {short_comment}")
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
        st.write(f"Inserted new embedding ID: {doc_id}")
    print("Timesheet database updated.")
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
# LLM-as-Judge: Evaluation function for the RAG system
# ----------------------------
def evaluate_rag_system(original_query, refined_query, retrieved_docs, final_answer, model="llama3.2"):
    """
    Uses the LLM as a judge to evaluate the RAG system.
    
    Parameters:
        original_query (str): The original user query.
        refined_query (str): The refined query generated by the system.
        retrieved_docs (list): List of strings representing the retrieved documents (each document's text).
        final_answer (str): The final answer provided by the RAG system.
        model (str): The LLM model to use (default: "llama3.2").
        
    Returns:
        dict: A dictionary containing scores and comments for each evaluation category.
              Expected keys: accuracy_score, accuracy_comments, relevance_score, relevance_comments,
              clarity_score, clarity_comments, overall_score, overall_comments.
    """
    # Format the retrieved documents as bullet points
    formatted_docs = "\n".join([f"- {doc}" for doc in retrieved_docs])
    
    evaluation_prompt = (
        "You are an evaluator for a Retrieval Augmented Generation (RAG) system. "
        "Below are the outputs from different stages of the system:\n\n"
        f"**Original Query:** {original_query}\n\n"
        f"**Refined Query:** {refined_query}\n\n"
        f"**Retrieved Documents/Context:**\n{formatted_docs}\n\n"
        f"**Final Answer:** {final_answer}\n\n"
        "Please evaluate the performance of the RAG system based on the following criteria:\n"
        "1. Accuracy: How accurately does the final answer address the original query?\n"
        "2. Relevance: How relevant are the retrieved documents to the query?\n"
        "3. Clarity: Is the refined query clear and does it capture the user’s intent?\n"
        "4. Overall Performance: What is the overall quality of the system?\n\n"
        "For each category, provide detailed feedback and assign a score between 1 (poor) and 10 (excellent). "
        "Return your response as a JSON object with the following structure:\n\n"
        "{\n"
        '  "accuracy_score": <score>,\n'
        '  "accuracy_comments": "Your comments here",\n'
        '  "relevance_score": <score>,\n'
        '  "relevance_comments": "Your comments here",\n'
        '  "clarity_score": <score>,\n'
        '  "clarity_comments": "Your comments here",\n'
        '  "overall_score": <score>,\n'
        '  "overall_comments": "Your comments here"\n'
        "}"
    )
    
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": evaluation_prompt}],
    )
    
    result = response["message"]["content"]
    try:
        evaluation = json.loads(result)
    except Exception as e:
        evaluation = {"error": f"Could not parse LLM response: {result}"}
    return evaluation

# ----------------------------
# Streamlit App Interface
# ----------------------------
def main():
    st.title("Timesheet LLM & ChromaDB Demo with RAG Evaluation")

    # Section to update/build the timesheet DB
    if st.button("Update Timesheet Database"):
        try:
            df = pd.read_excel(TIMESHEET_FILEPATH)
        except Exception as e:
            st.error(f"Error loading Excel file: {e}")
            return

        collection = get_chroma_collection(CHROMADB_PATH, COLLECTION_NAME)
        with st.spinner("Processing timesheet data..."):
            generate_timesheet_db(df, collection)
        st.success("Database updated successfully!")

    st.markdown("---")

    # Section for query processing
    st.header("Query the Timesheet Database")
    original_query = st.text_input("Enter your query", value="")

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

        # ----------------------------
        # Evaluation Section: Using LLM as Judge
        # ----------------------------
        st.markdown("---")
        st.header("Evaluate RAG System")
        if st.button("Evaluate RAG System"):
            # For evaluation, we pass the original query, refined query, retrieved docs texts, and final answer.
            retrieved_texts = [d["text"] for d in results]
            with st.spinner("Evaluating RAG system..."):
                evaluation = evaluate_rag_system(original_query, refined_query, retrieved_texts, final_answer)
            
            st.subheader("Evaluation Results")
            if "error" in evaluation:
                st.error(evaluation["error"])
            else:
                st.metric("Accuracy Score", evaluation.get("accuracy_score", "N/A"))
                st.write("Accuracy Comments:", evaluation.get("accuracy_comments", "N/A"))
                st.metric("Relevance Score", evaluation.get("relevance_score", "N/A"))
                st.write("Relevance Comments:", evaluation.get("relevance_comments", "N/A"))
                st.metric("Clarity Score", evaluation.get("clarity_score", "N/A"))
                st.write("Clarity Comments:", evaluation.get("clarity_comments", "N/A"))
                st.metric("Overall Score", evaluation.get("overall_score", "N/A"))
                st.write("Overall Comments:", evaluation.get("overall_comments", "N/A"))
                with st.expander("Show Raw Evaluation JSON"):
                    st.json(evaluation)

if __name__ == "__main__":
    main()
