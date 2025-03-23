import gradio as gr
import whisper
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import os
import tempfile


os.environ["PATH"] += os.pathsep + "C:\\Users\\ADMIN\\Downloads\\ffmpeg-N-118896-g9f0970ee35-win64-gpl-shared\\ffmpeg-N-118896-g9f0970ee35-win64-gpl-shared\\bin"

# ---- Global Variables ----
DB_PATH = "./embeddings"
MODEL_NAME = "all-MiniLM-L6-v2"

# ---- Initialize DB & Embedding Model ----
def init_chroma():
    client = chromadb.Client()
    db = client.create_collection(name="subtitles")
    return client, db

def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)

client, db = init_chroma()
embed_model = load_embedding_model()

# ---- Function: Transcribe Audio ----
def transcribe_audio(file_path):
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result["text"]

# ---- Function: Preprocess Subtitle Documents ----
def load_subtitle_docs(folder_path="./database"):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                docs.append(f.read())
    return docs

# ---- Function: Chunking for Embeddings ----
def chunk_text(text, chunk_size=500, overlap=100):
    tokens = text.split()
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = " ".join(tokens[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# ---- Function: Embed Documents and Store in ChromaDB ----
def store_documents():
    docs = load_subtitle_docs()
    
    for idx, doc in enumerate(docs):
        chunks = chunk_text(doc)
        embeddings = embed_model.encode(chunks)

        for i, emb in enumerate(embeddings):
            db.add(
                documents=[chunks[i]],
                metadatas=[{"source": f"doc_{idx}_chunk_{i}"}],
                ids=[f"{idx}_{i}"]
            )

    return "Documents embedded successfully!"

# ---- Function: Search Subtitles ----
def search_subtitles(query, top_k=5):
    query_embedding = embed_model.encode([query])[0]

    results = db.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    # Display top results
    output = []
    if "documents" in results and results["documents"]:
        for result in results['documents'][0]:
            output.append(f"Match: {result}")
    else:
        output.append("No matches found.")
    
    return "\n".join(output)

# ---- Function: Process Audio & Search ----
def process_audio(file_path):
    """
    Handles both file paths and file-like objects from Gradio.
    """
    if not os.path.isfile(file_path):
        return "Error: File not found.", "No results."

    # Transcribe the audio
    transcription = transcribe_audio(file_path)

    # Search for matching subtitles
    results = search_subtitles(transcription)

    return transcription, results

# ---- Gradio Interface ----
def handle_gradio_file(file):
    """
    Ensure proper handling of Gradio's temporary file format.
    """
    if isinstance(file, str) and os.path.isfile(file):
        return process_audio(file)

    # Handle file-like object
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name

    return process_audio(tmp_file_path)

# ---- Gradio UI ----
with gr.Blocks() as app:
    gr.Markdown("# 🎵 Shazam Clone: Audio Transcription & Subtitle Search")

    with gr.Row():
        audio_input = gr.Audio(label="Upload Audio", type="filepath")
        output_transcript = gr.Textbox(label="Transcription", interactive=False)
        output_results = gr.Textbox(label="Search Results", interactive=False)
        
    audio_button = gr.Button("Transcribe & Search")
    
    audio_button.click(
        fn=handle_gradio_file,
        inputs=[audio_input],
        outputs=[output_transcript, output_results]
    )

app.launch(share=True)

