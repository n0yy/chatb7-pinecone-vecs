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

# Konfigurasi untuk menghindari masalah dengan torch.__path__._path
# Ini akan mencegah Streamlit mencoba menggunakan torch._classes yang menyebabkan error
import sys

if "torch._classes" in sys.modules:
    # Patch untuk mengatasi masalah dengan torch._classes.__path__
    torch_classes = sys.modules["torch._classes"]
    if not hasattr(torch_classes, "__path__"):
        setattr(torch_classes, "__path__", [])

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
    # Gunakan model_copy sebagai pengganti copy (untuk mengatasi peringatan Pydantic v2.0)
    try:
        if hasattr(doc, "model_copy"):
            # Untuk Pydantic v2.0+
            cleaned_doc = doc.model_copy()
        else:
            # Fallback untuk Pydantic v1.x
            cleaned_doc = doc.copy()
    except Exception as e:
        st.warning(f"Peringatan saat menyalin dokumen: {e}")
        cleaned_doc = doc  # Gunakan dokumen asli jika kedua metode gagal

    # Jika metadata memiliki field dl_meta, konversi ke string
    if (
        hasattr(cleaned_doc, "metadata")
        and cleaned_doc.metadata
        and "dl_meta" in cleaned_doc.metadata
    ):
        # Konversi objek kompleks menjadi string JSON
        try:
            cleaned_doc.metadata["dl_meta"] = json.dumps(
                cleaned_doc.metadata["dl_meta"]
            )
        except Exception as e:
            st.warning(f"Gagal mengkonversi dl_meta ke JSON: {e}")
            # Hapus field dl_meta jika tidak dapat dikonversi
            cleaned_doc.metadata.pop("dl_meta", None)

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

        # Tampilkan informasi ukuran chunk
        chunk_size = 1500
        chunk_overlap = 400
        st.info(
            f"Menggunakan chunk_size={chunk_size} dan chunk_overlap={chunk_overlap} untuk menghindari masalah batas token"
        )

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

                # Split the document - menggunakan chunk yang lebih kecil untuk mengatasi masalah token
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,  # Gunakan len() sebagai fungsi pengukur panjang default
                )
                docs = splitter.split_documents(doc)

                if not docs:
                    st.warning(
                        f"Peringatan: Tidak ada chunk yang dihasilkan dari file {file.name}"
                    )
                    continue

                # Bersihkan metadata pada setiap dokumen
                cleaned_docs = []
                for d in docs:
                    try:
                        cleaned_doc = clean_metadata(d)
                        cleaned_docs.append(cleaned_doc)
                    except Exception as e:
                        st.warning(
                            f"Gagal membersihkan metadata pada satu chunk dari {file.name}: {e}"
                        )

                if cleaned_docs:
                    all_docs.extend(cleaned_docs)
                    uuids.extend([str(uuid4()) for _ in range(len(cleaned_docs))])
                    success_count += 1
                    st.success(
                        f"Berhasil memproses {file.name}: {len(cleaned_docs)} chunk dihasilkan"
                    )
                else:
                    st.error(f"Tidak ada chunk yang valid dari file {file.name}")
                    failed_count += 1
            except Exception as e:
                failed_count += 1
                st.error(f"Error saat memproses file {file.name}: {e}")
            finally:
                # Hapus temp file
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    st.warning(f"Tidak dapat menghapus file temporary: {e}")

                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))

        if all_docs:
            with st.spinner("Menambahkan dokumen ke vector store..."):
                try:
                    start_time = time.time()
                    batch_size = 10  # Batasi jumlah dokumen yang diproses per batch
                    total_batches = (
                        len(all_docs) + batch_size - 1
                    ) // batch_size  # Hitung total batch

                    for batch_idx in range(total_batches):
                        # Hitung indeks awal dan akhir untuk batch saat ini
                        start_idx = batch_idx * batch_size
                        end_idx = min((batch_idx + 1) * batch_size, len(all_docs))
                        batch_docs = all_docs[start_idx:end_idx]
                        batch_uuids = uuids[start_idx:end_idx]

                        try:
                            # Tambahkan dokumen sebagai batch
                            vector_store.add_documents(
                                documents=batch_docs, ids=batch_uuids
                            )
                            status_text.text(
                                f"Menambahkan batch {batch_idx+1}/{total_batches} ke vector store"
                            )
                            progress_bar.progress((batch_idx + 1) / total_batches)
                        except Exception as e:
                            st.error(f"Error saat menambahkan batch {batch_idx+1}: {e}")
                            # Coba tambahkan dokumen satu per satu jika batch gagal
                            st.warning("Mencoba menambahkan dokumen satu per satu...")
                            for i, (doc, uuid) in enumerate(
                                zip(batch_docs, batch_uuids)
                            ):
                                try:
                                    vector_store.add_documents(
                                        documents=[doc], ids=[uuid]
                                    )
                                    status_text.text(
                                        f"Menambahkan dokumen {start_idx+i+1}/{len(all_docs)}"
                                    )
                                except Exception as e_single:
                                    st.error(
                                        f"Gagal menambahkan dokumen {start_idx+i+1}: {e_single}"
                                    )

                    end_time = time.time()
                    st.success(f"✅ Berhasil menambahkan dokumen ke Pinecone!")
                    st.info(f"Waktu proses: {(end_time - start_time):.2f} detik")
                except Exception as e:
                    st.error(f"Gagal menambahkan dokumen ke Pinecone: {e}")

            # Tampilkan ringkasan
            st.write(f"**Ringkasan:**")
            st.write(f"- File berhasil diproses: {success_count}")
            st.write(f"- File gagal diproses: {failed_count}")
            st.write(f"- Total dokumen ditambahkan: {len(all_docs)}")
        elif success_count == 0:
            st.warning("Tidak ada dokumen yang berhasil diproses.")
