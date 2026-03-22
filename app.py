import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Page Config (Prevents some UI glitches)
st.set_page_config(page_title="YouTube Q&A", layout="wide")

# 2. Setup API Key 
# On your local machine, you can type this in. On GitHub, use Streamlit Secrets.
hf_token = st.sidebar.text_input("Hugging Face Token", type="password")

st.title("🤖 YouTube Transcript Assistant")

video_id = st.text_input("Enter YouTube Video ID (e.g., dQw4w9WgXcQ)")

if video_id and hf_token:
    try:
        # Load Transcript
        with st.spinner("Fetching transcript..."):
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            transcript = " ".join([c["text"] for c in transcript_list])
        
        # Split and Embed
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.create_documents([transcript])
        
        # Use a lightweight embedding model
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(docs, embeddings)
        retriever = vector_store.as_retriever()

        # 3. Use the Inference API (No local download = No blank screen!)
        llm = HuggingFaceEndpoint(
            repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            huggingfacehub_api_token=hf_token,
            temperature=0.5
        )

        template = """Answer the question based only on the following context:
        {context}
        
        Question: {question}
        """
        prompt = PromptTemplate.from_template(template)

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        query = st.text_input("What would you like to know about the video?")
        if query:
            res = chain.invoke(query)
            st.info(res)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
        import traceback
