from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ------------------------------------
# CREATE VECTOR DATABASE
# ------------------------------------

def create_vector_db(transcript_text):

    # Split transcript into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(transcript_text)

    # Embedding Model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create Vector DB
    vector_db = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model
    )

    return vector_db


# ------------------------------------
# SEARCH RELEVANT CONTEXT
# ------------------------------------

def get_relevant_context(vector_db, query):

    docs = vector_db.similarity_search(query, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    return context