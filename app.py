import streamlit as st
import os
import re

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ================= PAGE =================
st.set_page_config(page_title="YouTube RAG Assistant", layout="wide")
st.title("🤖 YouTube Transcript Assistant")


# ================= TOKEN =================
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# ================= SESSION =================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


# ================= HELPERS =================
def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([^&]+)", url)
    return match.group(1) if match else None


@st.cache_resource
def build_vector_store(transcript_text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    docs = splitter.create_documents([transcript_text])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(docs, embeddings)


# ================= SIDEBAR =================
url = st.sidebar.text_input("🎬 Paste YouTube Video URL")

if st.sidebar.button("Load Video"):

    video_id = extract_video_id(url)

    if not video_id:
        st.sidebar.error("Invalid URL")
    else:
        try:
            with st.spinner("Fetching transcript and building knowledge base..."):
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                text = " ".join([t["text"] for t in transcript])

                st.session_state.vector_store = build_vector_store(text)
                st.session_state.messages = []

            st.sidebar.success("Video Loaded Successfully ✅")

        except Exception as e:
            st.sidebar.error(f"Transcript load failed: {e}")


# ================= CHAT =================
if st.session_state.vector_store and hf_token:

    retriever = st.session_state.vector_store.as_retriever()

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.2-3B-Instruct",
        huggingfacehub_api_token=hf_token,
        temperature=0.5,
        max_new_tokens=512
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_query := st.chat_input("Ask question about this video"):

        st.session_state.messages.append(
            {"role": "user", "content": user_query}
        )

        docs = retriever.invoke(user_query)
        context = "\n".join([d.page_content for d in docs])

        final_prompt = f"""
Answer ONLY using this transcript context:

{context}

Question:
{user_query}
"""

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = llm.invoke(final_prompt)
                st.markdown(response)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

else:
    st.info("👈 Paste YouTube URL and click Load Video")
