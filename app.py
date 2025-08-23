import streamlit as st
import psycopg2
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import ast
from collections import deque, defaultdict
import time
import dotenv
import nest_asyncio
import string, time

dotenv.load_dotenv()

# Apply at the beginning of your script
nest_asyncio.apply()

# --------------------------
# DB & Embeddings setup
# --------------------------
def get_db_connection():
    return psycopg2.connect(
        dbname="dsa_search",
        user="dsa_user",
        password="dsa_user",
        host="localhost",
        port="5432",
    )


# Then you can initialize the model at module level
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


# --------------------------
# Cosine similarity
# --------------------------
def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# --------------------------
# Knowledge Graph BFS
# --------------------------
def bfs_traverse(graph, start_id, max_depth):
    visited = set()
    queue = deque([(start_id, 0)])
    related_ids = set()

    while queue:
        node, depth = queue.popleft()
        if node in visited or depth > max_depth:
            continue
        visited.add(node)
        related_ids.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                queue.append((neighbor, depth + 1))

    related_ids.discard(start_id)
    return related_ids


def build_graph_from_db(threshold=0.8):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT source_id, target_id, score
        FROM dsa_question_edges_gemini
        WHERE score >= %s;
        """,
        (threshold,),
    )
    edges = cur.fetchall()
    cur.close()
    conn.close()

    graph = defaultdict(list)
    for source, target, score in edges:
        graph[source].append(target)
    return graph


def search_questions(query_text, top_k, bfs_max_depth):
    if not query_text.strip():
        return []

    # 1. Generate query embedding
    query_emb = embeddings_model.embed_query(query_text)

    # 2. Use database to find top-k similar questions
    conn = get_db_connection()
    cur = conn.cursor()

    # Use pgvector's cosine_similarity function instead of <=> operator
    cur.execute(
        """
        SELECT id, title, difficulty, topic, link, description, embedding, 
               1 - cosine_distance(embedding, %s::vector) as similarity
        FROM dsa_questions_gemini
        ORDER BY similarity DESC
        LIMIT %s;
        """,
        (query_emb, top_k),
    )

    top_k_rows = cur.fetchall()

    # 3. Process top-k results
    id_to_question = {}
    top_k_ids = []
    top_k_questions = []

    for row in top_k_rows:
        q_id, title, difficulty, topic, link, description, emb_str, similarity = row

        id_to_question[q_id] = {
            "title": title,
            "difficulty": difficulty,
            "topic": topic,
            "link": link,
            "description": description,
            "similarity": similarity,
        }

        top_k_ids.append(q_id)
        top_k_questions.append(id_to_question[q_id])

    # 4. BFS traversal for related questions - optimized to fetch only what's needed
    graph = build_graph_from_db(threshold=0.8)
    all_related_ids = set()

    for q_id in top_k_ids:
        related_ids = bfs_traverse(graph, q_id, max_depth=bfs_max_depth)
        all_related_ids.update(related_ids)

    # 5. Only fetch related questions that weren't in the top-k
    all_related_ids -= set(top_k_ids)

    if all_related_ids:
        # Fetch only the needed related questions
        placeholders = ",".join(["%s"] * len(all_related_ids))
        cur.execute(
            f"""
            SELECT id, title, difficulty, topic, link, description
            FROM dsa_questions_gemini
            WHERE id IN ({placeholders});
            """,
            tuple(all_related_ids),
        )

        related_rows = cur.fetchall()

        for row in related_rows:
            q_id, title, difficulty, topic, link, description = row
            top_k_questions.append(
                {
                    "title": title,
                    "difficulty": difficulty,
                    "topic": topic,
                    "link": link,
                    "description": description,
                    "similarity": None,  # Mark as related, not from similarity
                }
            )

    cur.close()
    conn.close()

    return top_k_questions


# --------------------------
# preprocessing
# --------------------------
def preprocessing(text):
    string.punctuation = string.punctuation.replace("%", "")
    string.punctuation = string.punctuation.replace('"', "•")
    string.punctuation = string.punctuation.replace(".", "")
    string.punctuation = string.punctuation.replace("@", "")
    for char in string.punctuation:
        text = text.replace(char, " ")
    text = " ".join(text.split())
    text = text.lower()
    text = text.strip()
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")
    return text


# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="DSA Search Engine", layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>🔥 DSA Question Search Engine</h1>",
    unsafe_allow_html=True,
)

query = st.text_input("Search your question:", value="", placeholder="Enter your question here...")

if query:
    with st.spinner("Searching for related questions... 🔍"):
        results = search_questions(query, top_k=2, bfs_max_depth=1)
        time.sleep(0.5)

    if results:
        for res in results:
            # Check if similarity score exists
            similarity_info = ""
            if res['similarity'] is not None:
              similarity_percentage = f"{res['similarity'] * 100:.2f}%"
              similarity_info = f"<strong>Similarity:</strong> {similarity_percentage} | "
            else:
              similarity_info = "<strong>Related Question</strong> | "
              
            st.markdown(
              f"""
              <div style='border:1px solid #ddd; padding:15px; border-radius:8px; margin-bottom:15px;'>
                <a href="{res['link']}" target="_blank" style='text-decoration:none; font-size:18px; font-weight:bold;'>{res['title']}</a>
                <p>{similarity_info}<strong>Difficulty:</strong> {res['difficulty']} | <strong>Topics:</strong> {res['topic']}</p>
                <p>{res['description']}</p>
              </div>
              """,
              unsafe_allow_html=True,
            )
    else:
        st.warning("No related questions found!")
