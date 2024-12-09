import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_csv_agent
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def initialize_groq():
    """Initialize Groq dengan API key"""
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    if not os.environ["GROQ_API_KEY"]:
        st.error("Groq API key tidak ditemukan di file .env!")
        st.stop()
    return ChatGroq(model="mixtral-8x7b-32768",temperature=0,max_tokens=None,timeout=None,max_retries=2,)

def load_data(uploaded_file):
    """Load data dari file yang diunggah"""
    filenames = []
    try:
        if uploaded_file.name.endswith(".csv"):
            filename = os.path.basename(uploaded_file.name)
            save_path = os.path.join("D:\MyPrograms\Langchain_Groqcloud_for_csv", filename)  # Ganti dengan path yang diinginkan
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())  # Menyimpan konten file
            filenames.append(filename)
            return filenames
        else:
            st.error("Format file tidak didukung. Unggah file dengan format CSV.")
            st.stop()
    except Exception as e:
        st.error(f"Gagal memuat file: {str(e)}")
        st.stop()

def file_to_dataframe(uploaded_file):
    """Mengubah file menjadi dataframe"""
    try:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Gagal memuat dataframe: {str(e)}")
        st.stop()

def process_query(agent, query):
    """Proses query dari user menggunakan PandasAI"""
    try:
        with st.spinner("🤖 Sedang menganalisis..."):
            response = agent.run(query)
            st.write("Jawaban:")
            st.write(response)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses pertanyaan: {str(e)}")

def main():
    st.set_page_config(
        page_title="Chatbot File CSV dengan Langchain dan Groq",
        page_icon="📊"
    )
    
    st.title("📊 Analisis File CSV dengan Langchain dan Groq")
    
    # Initialize OpenAI
    llm = initialize_groq()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Unggah File CSV", 
        type=["csv"]
    )
    
    if uploaded_file is not None:
        # Load data
        filenames = load_data(uploaded_file)
        
        # Show success message
        st.success(f"File '{uploaded_file.name}' berhasil dimuat. Data siap untuk dianalisis.")
        
        # Show dataframe preview
        df = file_to_dataframe(uploaded_file)
        st.subheader("Preview Data:")
        st.dataframe(df.head())
        
        # TODO : Extract filename into a list named csv_filenames, then input into create_csv_agent

        # Initialize Agent
        agent = create_csv_agent(
               llm,
               path = filenames,
               verbose=True,
               agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
               allow_dangerous_code = True)
            
        
        # User input
        user_input = st.text_area(
            "Tanyakan sesuatu tentang data Anda...",
            placeholder="Contoh: Berapa rata-rata nilai kolom X? Atau analisis tren dari data ini..."
        )
        
        # Process query when button is clicked
        if st.button("Kirim", type="primary"):
            if user_input:
                process_query(agent, user_input)
            else:
                st.warning("Harap masukkan pertanyaan terlebih dahulu.")

if __name__ == "__main__":
    main()