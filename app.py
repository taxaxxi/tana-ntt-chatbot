# ==================== app.py - Tana NTT Chatbot ====================
# 
# Chatbot untuk melestarikan pengetahuan pertanian leluhur NTT
# Dibangun dengan Streamlit + LangChain + Groq (GPT-OSS-120B)
#
# ==================================================================

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import tempfile

# ==================================================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# ==================================================================
st.set_page_config(
    page_title="Tana NTT - Pengetahuan Pertanian Leluhur",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Tana NTT")
st.caption("Asisten Digital untuk Melestarikan Pengetahuan Pertanian Leluhur Nusa Tenggara Timur")
st.markdown("---")

# ==================================================================
# 2. AMBIL API KEY GROQ
# ==================================================================
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("⚠️ API Key Groq tidak ditemukan.")
        st.info("""
        **Cara mengatasi:**
        1. Jika di lokal: `export GROQ_API_KEY=gsk_baru_anda` (Mac/Linux) 
           atau `set GROQ_API_KEY=gsk_baru_anda` (Windows)
        2. Jika di Streamlit Cloud: tambahkan di Settings → Secrets
        """)
        st.stop()

# ==================================================================
# 3. SIDEBAR - INFORMASI DAN UPLOAD PDF
# ==================================================================
with st.sidebar:
    st.header("🌾 Tentang Tana NTT")
    st.markdown("""
    **Tana NTT** adalah asisten digital yang bertugas melestarikan 
    dan membagikan pengetahuan pertanian lokal Nusa Tenggara Timur.
    
    **Yang Bisa Ditanyakan:**
    - ✅ Sistem pertanian tradisional (Mamar, Salome, Lodok, Kaliwu)
    - ✅ Ritual dan kalender pertanian (Halaika, Neon Oe, Hang Woja)
    - ✅ Tanaman pangan lokal (Sorgum, Jewawut, Uwi, dll)
    - ✅ Filosofi dan nilai (Ume Kbubu, Voe, Sen)
    """)
    
    st.divider()
    
    st.header("📄 Upload Pengetahuan")
    uploaded_file = st.file_uploader(
        "Upload file PDF pengetahuan Anda (opsional)",
        type="pdf",
        help="Upload PDF berisi pengetahuan pertanian lokal NTT"
    )
    st.caption("Jika tidak upload, akan menggunakan Knowledge Base bawaan.")

# ==================================================================
# 4. FUNGSI MEMBUAT RAG CHAIN (DICACHE UNTUK PERFORMANCE)
# ==================================================================
@st.cache_resource
def load_rag_chain(pdf_file=None):
    """
    Membuat RAG chain dari PDF yang diupload atau file bawaan.
    Hasilnya di-cache agar tidak reload setiap kali user chat.
    """
    
    if pdf_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            temp_path = tmp_file.name
        pdf_path = temp_path
        st.sidebar.success("✅ Menggunakan file yang diupload!")
    else:
        if os.path.exists("Tana_NTT_Knowledge.pdf"):
            pdf_path = "Tana_NTT_Knowledge.pdf"
            st.sidebar.info("📖 Menggunakan Knowledge Base bawaan")
        else:
            st.sidebar.error("❌ Tidak ada file PDF! Upload file di sidebar.")
            return None

    with st.spinner("📄 Membaca file PDF..."):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

    with st.spinner("✂️ Memotong dokumen menjadi bagian-bagian kecil..."):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(documents)
        st.sidebar.write(f"📊 {len(docs)} potongan dokumen siap")

    with st.spinner("🧠 Membuat vector database..."):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(docs, embeddings)
        st.sidebar.success("✅ Vector database siap!")

    with st.spinner("🤖 Menghubungkan ke model AI..."):
        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.7,
            max_tokens=512,
            api_key=groq_api_key
        )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    template = """
Anda adalah "Tana NTT", asisten digital yang bertugas melestarikan dan membagikan 
pengetahuan pertanian lokal Nusa Tenggara Timur. Anda berbicara dengan gaya seperti 
penutur cerita (storyteller) yang bijaksana, penuh hormat terhadap tradisi dan leluhur.

Tugas Anda:
1. Menjawab pertanyaan tentang pertanian tradisional NTT berdasarkan konteks yang diberikan.
2. Menyebutkan asal-usul pengetahuan (misal: "Menurut tradisi Suku Boti...", 
   "Dalam ritus pertanian Dawan...").
3. Menghubungkan pengetahuan tradisional dengan konsep ilmu tanah modern jika relevan.
4. Jika konteks tidak cukup, jawab dengan hormat bahwa Anda belum mengetahuinya dan 
   sarankan untuk berkonsultasi dengan tetua adat atau sumber lain.
5. Gunakan istilah lokal seperti: Ume Kbubu (lumbung), Neon Oe (hari air), 
   Halaika (filosofi alam), Voe (gotong royong), Sen (satu/bersama), 
   Salome (sistem tanam), Mamar (agroforestri), Lodok (sawah jaring laba-laba).

SELALU akhiri jawaban Anda dengan pesan pelestarian, seperti:
"Pengetahuan ini adalah warisan leluhur yang telah terbukti menjaga ketahanan pangan 
selama bergenerasi. Mari kita jaga dan wariskan kepada generasi berikutnya."

Konteks:
{context}

Pertanyaan pengguna: {question}

Jawaban Tana NTT yang bijaksana dan penuh kearifan:
"""

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    st.sidebar.success("✅ Tana NTT siap berbicara!")
    return rag_chain

# ==================================================================
# 5. INISIALISASI RAG CHAIN
# ==================================================================
rag_chain = load_rag_chain(uploaded_file)

# ==================================================================
# 6. LOGIKA CHATBOT (DENGAN MEMORY)
# ==================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Tanyakan tentang kearifan pertanian NTT..."):
    if rag_chain is None:
        st.warning("⚠️ Silakan upload file PDF terlebih dahulu di sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🌾 Sedang merenung dan menggali kearifan..."):
            try:
                answer = rag_chain.invoke(prompt)
                st.markdown(answer)
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {e}")
                answer = "Maaf, saya mengalami kesulitan. Silakan coba lagi nanti."

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ==================================================================
# 7. FOOTER
# ==================================================================
st.markdown("---")
st.caption("🌾 Tana NTT - Melestarikan Pengetahuan Pertanian Leluhur NTT")
