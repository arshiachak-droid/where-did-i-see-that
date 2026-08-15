import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pypdf import PdfReader
import os

# 📊 1. CUSTOMIZE BROWSER TAB AND LOGO
st.set_page_config(
    page_title="Where Did I See That?",
    page_icon="🔍",
    layout="centered"
)


# 2. Initialize the AI Model and cache it globally
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L3-v2')


model = load_model()

# 📂 3. AUTO-DATABASE STORAGE FOLDER
# This permanently holds the files behind the scenes so they never vanish
DB_FOLDER = "user_permanent_database"
if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)


# 4. Helper function to read all stored data from disk
def load_stored_database():
    docs = []
    names = []
    for filename in os.listdir(DB_FOLDER):
        file_path = os.path.join(DB_FOLDER, filename)
        if filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                docs.append(f.read())
                names.append(filename)
        elif filename.endswith(".pdf"):
            try:
                reader = PdfReader(file_path)
                pdf_text = ""
                for page in reader.pages:
                    pdf_text += page.extract_text() + "\n"
                if pdf_text.strip():
                    docs.append(pdf_text)
                    names.append(filename)
            except Exception as e:
                pass
    return docs, names


# 5. Web UI Header Elements
st.title("🔍 'Where Did I See That?'")
st.subheader("Your AI-Powered Local Brain Dump")

# 📥 6. EASY DRAG & DROP INTERFACE
uploaded_files = st.file_uploader(
    "Add new documents to your permanent AI memory (.txt or .pdf):",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

# 💾 7. AUTOMATIC CLOUD-STYLE SAVING
if uploaded_files:
    new_saved = False
    for uploaded_file in uploaded_files:
        file_path = os.path.join(DB_FOLDER, uploaded_file.name)
        # Only save if it doesn't already exist on disk
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            new_saved = True

    if new_saved:
        st.success("✨ New documents merged into your permanent memory database!")
        st.rerun()

# 8. Load the master database of files
documents, source_names = load_stored_database()

# 9. Run Search Logic if the database contains files
if documents:
    st.markdown("---")
    st.success(f"📦 AI Memory Status: Active ({len(documents)} document(s) permanently saved)")

    # Calculate vectors on the active database
    with st.spinner("AI formatting database coordinates..."):
        doc_embeddings = model.encode(documents)
        dimension = doc_embeddings.shape[1]  # Get exact vector size length

        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(doc_embeddings).astype('float32'))

    # Text search box
    query = st.text_input("Search your brain dump:", placeholder="Type anything you want to find...")

    if query:
        query_embedding = model.encode([query])
        k = min(2, len(documents))
        distances, indices = index.search(np.array(query_embedding).astype('float32'), k)

        st.write("### 🤖 Best Matches Found:")
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue

            confidence = float(1 / (1 + distances[0][i]))

            with st.expander(f"📄 {source_names[idx]} (Match: {confidence:.2%})", expanded=True):
                raw_text = documents[idx]
                snippet = raw_text[:500] + "..." if len(raw_text) > 500 else raw_text
                st.write(snippet)

    # 🗑️ EXTRA CREDIT: Allow users to clear their database right from the web UI
    if st.button("🗑️ Clear Entire Permanent Memory"):
        for filename in os.listdir(DB_FOLDER):
            os.remove(os.path.join(DB_FOLDER, filename))
        st.warning("All files deleted from permanent memory.")
        st.rerun()
else:
    st.info(
        "💡 Your AI database is completely empty. Drag and drop a file above to permanently save it to your dashboard!")
