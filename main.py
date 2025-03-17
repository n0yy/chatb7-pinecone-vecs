import streamlit as st
import os
import tempfile
import time
import json
from langchain_docling import DoclingLoader
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from uuid import uuid4

# Setup
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector_store = PineconeVectorStore(index_name="chatb7", embedding=embeddings)

# UI
st.title("ChatB7 Vector Store")

uploaded_files = st.file_uploader(
    "Upload files", type=["pdf", "docx", "pptx"], accept_multiple_files=True
)
submit = st.button("Submit")


def clean_metadata(doc):
    """Membersihkan metadata agar sesuai dengan format Pinecone."""
    # Salin dokumen untuk menghindari modifikasi asli
    cleaned_doc = doc.copy()

    # Jika metadata memiliki field dl_meta, konversi ke string
    if hasattr(cleaned_doc, "metadata") and "dl_meta" in cleaned_doc.metadata:
        # Konversi objek kompleks menjadi string JSON
        cleaned_doc.metadata["dl_meta"] = json.dumps(cleaned_doc.metadata["dl_meta"])

    # Jika ada metadata lain yang kompleks, bisa ditambahkan penanganan di sini
    return cleaned_doc


if submit:
    if not uploaded_files:
        st.warning("Harap unggah setidaknya satu file terlebih dahulu.")
    else:
        all_docs = []
        uuids = []
        success_count = 0
        failed_count = 0
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, file in enumerate(uploaded_files):
            status_text.text(f"Memproses file {i+1}/{len(uploaded_files)}: {file.name}")

            # Save the file to temporary directory
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.getvalue())
                temp_file_path = temp_file.name

            try:
                # Load the document
                loader = DoclingLoader(temp_file_path)
                doc = loader.load()

                # Split the document
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                docs = splitter.split_documents(doc)

                # Bersihkan metadata pada setiap dokumen
                cleaned_docs = [clean_metadata(d) for d in docs]

                all_docs.extend(cleaned_docs)
                uuids.extend([str(uuid4()) for _ in range(len(cleaned_docs))])
                success_count += 1
            except Exception as e:
                failed_count += 1
                st.error(f"Error saat memproses file {file.name}: {e}")
            finally:
                # Hapus temp file
                os.unlink(temp_file_path)

                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))

        if all_docs:
            with st.spinner("Menambahkan dokumen ke vector store..."):
                try:
                    start_time = time.time()
                    # Tambahkan dokumen satu per satu untuk menghindari error batch
                    for i, (doc, uuid) in enumerate(zip(all_docs, uuids)):
                        try:
                            vector_store.add_documents(documents=[doc], ids=[uuid])
                            status_text.text(
                                f"Menambahkan dokumen {i+1}/{len(all_docs)}"
                            )
                            progress_bar.progress((i + 1) / len(all_docs))
                        except Exception as e:
                            st.error(f"Error saat menambahkan dokumen {i+1}: {e}")

                    end_time = time.time()
                    st.success(f"✅ Berhasil menambahkan dokumen ke Pinecone!")
                    st.info(f"Waktu proses: {((end_time - start_time) / 60):.2f} detik")
                except Exception as e:
                    st.error(f"Gagal menambahkan dokumen ke Pinecone: {e}")

            # Tampilkan ringkasan
            st.write(f"**Ringkasan:**")
            st.write(f"- File berhasil diproses: {success_count}")
            st.write(f"- File gagal diproses: {failed_count}")
            st.write(f"- Total dokumen ditambahkan: {len(all_docs)}")
        elif success_count == 0:
            st.warning("Tidak ada dokumen yang berhasil diproses.")
