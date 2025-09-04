# 🔥 DSA Search Engine

A powerful search engine for Data Structures and Algorithms (DSA) questions built with advanced embedding techniques and graph-based similarity matching.

## 🚀 Features

- **Semantic Search**: Uses Google Generative AI embeddings for intelligent question matching
- **TF-IDF Search**: Traditional text-based search using TF-IDF vectorization
- **Graph-Based Recommendations**: BFS traversal through question similarity graphs
- **Real-time Results**: Fast search with PostgreSQL + pgvector optimization
- **Interactive UI**: Clean Streamlit interface for easy question discovery

## 🛠️ Tech Stack

- **Backend**: Python, PostgreSQL with pgvector extension
- **AI/ML**: Google Generative AI, scikit-learn, TF-IDF
- **Frontend**: Streamlit
- **Data Processing**: Beautiful Soup, Requests, Jupyter Notebooks
- **Database**: PostgreSQL with vector similarity search

## 📋 Prerequisites

- Python 3.8+
- PostgreSQL with pgvector extension
- Google AI API key
- Conda or pip for package management

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/PSPritish/dsa_search_engine.git
   cd dsa_search_engine
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up PostgreSQL with pgvector**
   ```bash
   # Install pgvector extension
   sudo apt-get install postgresql-14-pgvector
   ```

4. **Create database and user**
   ```sql
   -- Connect as postgres user
   sudo -u postgres psql
   
   -- Create database and user
   CREATE DATABASE dsa_search;
   CREATE USER dsa_user WITH PASSWORD 'dsa_user';
   GRANT ALL PRIVILEGES ON DATABASE dsa_search TO dsa_user;
   
   -- Connect to the database and enable pgvector
   \c dsa_search
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

5. **Set up environment variables**
   ```bash
   # Create .env file
   touch .env
   ```
   
   Add your configuration to `.env`:
   ```
   GOOGLE_API_KEY=your_google_ai_api_key_here
   DATABASE_URL=postgresql://dsa_user:dsa_user@localhost:5432/dsa_search
   ```

## 📊 Database Schema

The project uses several tables for storing questions and similarity relationships:

- `dsa_questions_gemini`: Stores questions with Google AI embeddings
- `dsa_questions_tfidf`: Stores questions with TF-IDF embeddings  
- `dsa_question_edges_gemini`: Stores similarity relationships between questions

## 🚀 Usage

### 1. Data Collection and Processing

Run the Jupyter notebook to scrape and process LeetCode questions:

```bash
jupyter lab webScrapping.ipynb
```

This will:
- Fetch questions from LeetCode API
- Generate embeddings using Google AI
- Create TF-IDF vectors
- Build similarity graphs
- Store everything in PostgreSQL

### 2. Run the Gemini-based Search Engine

```bash
streamlit run app.py
```

### 3. Run the TF-IDF Search Engine

```bash
streamlit run tfidf.py
```

## 🔍 How It Works

### Semantic Search (Gemini)
1. User enters a query
2. Query is embedded using Google Generative AI
3. Vector similarity search finds top-k most similar questions
4. BFS traversal discovers related questions through the similarity graph
5. Results are ranked and displayed

### TF-IDF Search
1. User query is processed using the pre-trained TF-IDF vectorizer
2. Cosine similarity computed against all stored question vectors
3. Top results returned based on similarity scores

### Graph-Based Discovery
- Questions are connected in a graph based on content similarity
- BFS traversal helps discover related questions beyond direct matches
- Configurable depth and similarity thresholds

## 📁 Project Structure

```
dsa_search_engine/
├── app.py                    # Main Streamlit app (Gemini-based search)
├── tfidf.py                 # TF-IDF based search app
├── webScrapping.ipynb       # Data collection and processing notebook
├── requirements.txt         # Python dependencies
├── tfidf_vectorizer.pkl     # Saved TF-IDF model
├── .env                     # Environment variables
└── README.md               # This file
```

## ⚙️ Configuration

### Search Parameters

- `top_k`: Number of initial similar questions to retrieve (default: 10)
- `bfs_max_depth`: Maximum depth for graph traversal (default: 2)
- `threshold`: Minimum similarity score for graph edges (default: 0.8)

### Database Settings

Configure your database connection in `.env`:
```
DATABASE_URL=postgresql://username:password@host:port/database_name
```

## 🔧 API Integration

The project integrates with:
- **LeetCode GraphQL API**: For fetching question data
- **Google Generative AI API**: For generating semantic embeddings

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- LeetCode for providing the question data
- Google AI for the embedding model
- The open-source community for the amazing tools and libraries

## 📞 Contact

**Pritish** - [@PSPritish](https://github.com/PSPritish)

Project Link: [https://github.com/PSPritish/dsa_search_engine](https://github.com/PSPritish/dsa_search_engine)

---

⭐ Star this repository if you find it helpful!
