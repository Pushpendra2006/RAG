import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Page Config
st.set_page_config(page_title="YouTube RAG Assistant", layout="wide")
st.title("🤖 YouTube Transcript Assistant")

# 2. Accessing Streamlit Secrets
# This replaces os.getenv. In Streamlit Cloud, set this in 'Advanced Settings' -> 'Secrets'
hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    st.sidebar.warning("API Token not found in Secrets. Please add it to Settings or enter below:")
    hf_token = st.sidebar.text_input("Hugging Face Token", type="password")

# 3. Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Cached Function for Processing the Video
@st.cache_resource
def process_video(v_id):
    try:
        # Fetching Transcript
        loader = YouTubeTranscriptApi.get_transcript(v_id)
        text = " ".join([t["text"] for t in loader])
        
        # Splitting
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.create_documents([text])
        
        # Embedding
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.from_documents(docs, embeddings)
    except Exception as e:
        st.error(f"Could not load transcript: {e}")
        return None

# --- UI Layout ---
video_id = st.sidebar.text_input("YouTube Video ID", placeholder="e.g., dQw4w9WgXcQ")

if video_id and hf_token:
    vector_store = process_video(video_id)
    
    if vector_store:
        retriever = vector_store.as_retriever()

        # 5. Define the LLM (using the Secret token)
        llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.2-3B-Instruct",
            huggingfacehub_api_token=hf_token,
            temperature=0.7
        )

        # 6. Chat Prompt with History Support
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer based on the video transcript: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        # Formatting history for the LLM
        def get_chat_history(messages):
            return [(m["role"], m["content"]) for m in messages]

        # 7. The RAG Chain
        rag_chain = (
            {
                "context": retriever,
                "chat_history": lambda x: get_chat_history(st.session_state.messages),
                "question": RunnablePassthrough()
            }
            | qa_prompt
            | llm
            | StrOutputParser()
        )

        # 8. Display Message History
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # 9. Handle New User Input
        if user_query := st.chat_input("Ask a question about this video..."):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing transcript..."):
                    ai_response = rag_chain.invoke(user_query)
                    st.markdown(ai_response)
            
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
else:
    st.info("👈 Enter a YouTube Video ID and ensure your Hugging Face Token is set to start chatting.")