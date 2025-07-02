import os
import glob
from markdown import markdown
from bs4 import BeautifulSoup
from chromadb import PersistentClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()

docs_dir = "../ds-rpc-01/resources/data"


def md_to_text(md_path):
    with open(md_path,"r",encoding="utf-8") as file:
        html_conversion=markdown(file.read())
        text_coversion=BeautifulSoup(html_conversion,features="html.parser")
        return text_coversion.get_text()
    
md_files = glob.glob(os.path.join(docs_dir, "**/*.md"), recursive=True)
csv_files = glob.glob(os.path.join(docs_dir, "**/*.csv"), recursive=True) 
print(md_files)
print(csv_files)
def get_role_from_path(filepath):
    p = Path(filepath).resolve()
    base = Path(docs_dir).resolve()
    try:
        rel = p.relative_to(base)
        return rel.parts[0].lower()
    except ValueError:
        return "default"
all_chunks= []
metadata= []
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# For each doc:

for md_file in md_files:
    role = get_role_from_path(md_file)
    if role == "general":
        role = "employee"
    text = md_to_text(md_file)
    chunks = text_splitter.split_text(text)
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) > 20:
            all_chunks.append(chunk)
            metadata.append({"role": role,"source": os.path.basename(md_file)})

print(f" Loaded {len(all_chunks)} chunks ")



def get_openai_embeddings(texts):
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [item.embedding for item in response.data]
def batch_embed(texts, batch_size=100):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = get_openai_embeddings(batch)
        embeddings.extend(emb)
    return embeddings

print("Embedding chunks using OpenAI...")
embeddings = batch_embed(all_chunks)
chroma_client = PersistentClient(path="./chroma_db")


collection = chroma_client.get_or_create_collection(name="company_docs")

collection.add(
    embeddings=embeddings,
    documents=all_chunks,
    metadatas=metadata,
    ids=[f"doc_{i}" for i in range(len(all_chunks))]
)


