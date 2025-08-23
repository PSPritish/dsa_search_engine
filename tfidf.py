import streamlit as st
import psycopg2
import numpy as np
import json
from collections import deque, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import string
import time
import scipy.sparse as sp
import joblib 

# --------------------------
# DB connection
# --------------------------
def get_db_connection():
    return psycopg2.connect(
        dbname="dsa_search",
        user="dsa_user",
        password="dsa_user",
        host="localhost",
        port="5432",
    )

# --------------------------
# Preprocessing
# --------------------------
def preprocessing(text):
    string.punctuation = string.punctuation.replace("%", "")
    string.punctuation = string.punctuation.replace('"', "•")
    string.punctuation = string.punctuation.replace(".", "")
    string.punctuation = string.punctuation.replace("@", "")
    for char in string.punctuation:
        text = text.replace(char, " ")
    text = " ".join(text.split())
    text = text.lower().strip()
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return text

# --------------------------
# BFS for adjacency graph traversal
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

def build_graph_from_db(threshold=0.3):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT source_id, target_id, score
        FROM dsa_question_edges_tfidf
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

# Load the vectorizer once when the app starts
@st.cache_resource
def load_vectorizer():
    return joblib.load('tfidf_vectorizer.pkl')

vectorizer = load_vectorizer()

# --------------------------
# Search using TF-IDF embeddings
# --------------------------
def search_questions_tfidf(query_text, top_k=5, bfs_max_depth=1):
    if not query_text.strip():
        return []

    conn = get_db_connection()
    cur = conn.cursor()

    # Fetch all questions and TF-IDF embeddings
    cur.execute("SELECT id, title, description, difficulty, topic, link, embedding FROM dsa_questions_tfidf")
    rows = cur.fetchall()

    ids, embeddings = [], []
    for row in rows:
        ids.append(row[0])
        emb_dict = row[6]
        # Handle empty embeddings
        if not emb_dict:
            embeddings.append(np.zeros(vectorizer.max_features, dtype=np.float32))
            continue
        max_idx = max([int(k) for k in emb_dict.keys()]) + 1
        emb = np.zeros(max_idx, dtype=np.float32)
        for k, v in emb_dict.items():
            emb[int(k)] = v
        embeddings.append(emb)

    # Pad to same length (should be 3000)
    max_len = vectorizer.max_features
    embeddings = np.array([np.pad(e, (0, max_len - len(e))) for e in embeddings], dtype=np.float32)

    # Use the loaded vectorizer to transform the query
    query_sparse = vectorizer.transform([preprocessing(query_text)])
    query_vec = np.array(query_sparse.todense())[0]

    # Compute cosine similarity
    sims = cosine_similarity(query_vec.reshape(1, -1), embeddings)[0]
    top_k_idx = sims.argsort()[-top_k:][::-1]

    top_k_questions = []
    id_to_question = {row[0]: row for row in rows}

    for idx in top_k_idx:
        q_id = ids[idx]
        row = id_to_question[q_id]
        top_k_questions.append({
            "title": row[1],
            "description": row[2],
            "difficulty": row[3],
            "topic": row[4],
            "link": row[5],
            "similarity": float(sims[idx])
        })

    # BFS traversal for related questions
    graph = build_graph_from_db(threshold=0.3)
    all_related_ids = set()
    for idx in top_k_idx:
        related_ids = bfs_traverse(graph, ids[idx], max_depth=bfs_max_depth)
        all_related_ids.update(related_ids)
    all_related_ids -= set([ids[idx] for idx in top_k_idx])

    if all_related_ids:
        placeholders = ",".join(["%s"] * len(all_related_ids))
        cur.execute(f"""
            SELECT id, title, description, difficulty, topic, link
            FROM dsa_questions_tfidf
            WHERE id IN ({placeholders});
        """, tuple(all_related_ids))
        related_rows = cur.fetchall()
        for row in related_rows:
            top_k_questions.append({
                "title": row[1],
                "description": row[2],
                "difficulty": row[3],
                "topic": row[4],
                "link": row[5],
                "similarity": None
            })

    cur.close()
    conn.close()
    return top_k_questions

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="DSA TF-IDF Search", layout="wide")
st.markdown("<h1 style='text-align: center;'>📝 DSA TF-IDF Search Engine</h1>", unsafe_allow_html=True)

query = st.text_input("Search your question:", value="", placeholder="Enter your question here...")

if query:
    with st.spinner("Searching for TF-IDF related questions... 🔍"):
        results = search_questions_tfidf(query, top_k=2, bfs_max_depth=1)
        time.sleep(0.5)

    if results:
        for res in results:
            similarity_info = ""
            if res['similarity'] is not None:
                similarity_info = f"<strong>Similarity:</strong> {res['similarity']*100:.2f}% | "
            else:
                similarity_info = "<strong>Related Question</strong> | "

            st.markdown(f"""
              <div style='border:1px solid #ddd; padding:15px; border-radius:8px; margin-bottom:15px;'>
                <a href="{res['link']}" target="_blank" style='text-decoration:none; font-size:18px; font-weight:bold;'>{res['title']}</a>
                <p>{similarity_info}<strong>Difficulty:</strong> {res['difficulty']} | <strong>Topics:</strong> {res['topic']}</p>
                <p>{res['description']}</p>
              </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("No related questions found!")
