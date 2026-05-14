#!/usr/bin/env python3
"""Build FAISS index from Math_Project/ .txt files. Run: python create_index.py"""

import os
import sys
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(SCRIPT_DIR, "Math_Project")
INDEX_PATH = os.path.join(SCRIPT_DIR, "faiss_index")
EMB_MODEL = "all-MiniLM-L6-v2"

print("=" * 55)
print("  FAISS Index Builder")
print("=" * 55)

if not os.path.exists(KB_PATH):
    print(f"Error: '{KB_PATH}' not found."); sys.exit(1)

txt_files = sorted(f for f in os.listdir(KB_PATH) if f.endswith(".txt"))
print(f"\n  Files found: {len(txt_files)}")
for f in txt_files:
    print(f"    - {f}")

documents = []
for f in txt_files:
    documents.extend(TextLoader(os.path.join(KB_PATH, f)).load())

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)
print(f"\n  Chunks: {len(docs)}")

print("  Building embeddings...")
embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
db = FAISS.from_documents(docs, embeddings)

os.makedirs(INDEX_PATH, exist_ok=True)
db.save_local(INDEX_PATH)
print(f"\n  ✅ Index saved → {INDEX_PATH} ({db.index.ntotal} vectors)")
print("=" * 55)