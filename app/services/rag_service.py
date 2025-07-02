from chromadb import PersistentClient
from dotenv import load_dotenv
import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client_db = PersistentClient(path="./chroma_db")
collection = client_db.get_or_create_collection(name="company_docs")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
csv_folder = "resources/data"

# Dictionary to store agents for each role
csv_agents = {}

for dirpath, _, filenames in os.walk(csv_folder):
    for file in filenames:
        if file.endswith(".csv"):
            full_path = os.path.join(dirpath, file)
            role = os.path.relpath(full_path, csv_folder).split(os.sep)[0].lower()

            try:
                df = pd.read_csv(full_path)
                csv_agents[role] = create_pandas_dataframe_agent(llm, df, verbose=True,agent_type="openai-tools", 
                        max_iterations=10, 
                        early_stopping_method="generate",
                        allow_dangerous_code=True)
            except Exception as e:
                print(f"❌ Failed to load CSV agent for {role}: {e}")

def get_openai_embedding(text: str) -> list:
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"  
    )
    return response.data[0].embedding
role_mapping = {
    "employee": "general"  # Employee accesses general data
}

def rag_answer(query: str, role: str) -> str:
    print(f"Handling query for role: '{role}'")
    mapped_role = role_mapping.get(role, role)
    citations = ""
    if role in csv_agents:
        print(f"🔍 Using CSV agent for role: '{role}'")

        try:
            csv_response = csv_agents[role].run(query)
            context = f"CSV Agent Output:\n{csv_response}"
            role = os.path.relpath(full_path, csv_folder).split(os.sep)[0].lower()
            citations = f"\n\nSources:\n- **{role.capitalize()} Department CSV**\n"
            
        except Exception as e:
            return f"❌ Failed to run CSV agent for '{role}': {e}"

    else:
        print(f"🔍 Using ChromaDB RAG for role: '{role}'")

        # 1. Embed query
        query_embed = get_openai_embedding(query)

        if role == "c-level":
            results = collection.query(
                query_embeddings=[query_embed],
                n_results=10  # full access
            )
        else:
            results = collection.query(
                query_embeddings=[query_embed],
                n_results=10,
                where={"role": mapped_role}  # restrict to their department
            )

        # Handle if no documents are returned
        if not results["documents"] or not results["documents"][0]:
            return "I'm sorry, I couldn't find an exact answer based on your department's data."

        context = "\n".join(results["documents"][0])
        context_chunks = results["documents"][0]
        metas = results["metadatas"][0]  

    # Optional: deduplicate sources
        seen = set()
        citations = "\n\nSources:\n"
        for i, meta in enumerate(metas):
            source = source = meta.get("source", "Unknown File")
            if source not in seen:
                citations += f"- {source}\n"
                seen.add(source)

        context = "\n".join(context_chunks)
    
    # Build the prompt using the context
    prompt = f"""
You are a helpful enterprise assistant that provides accurate answers only using the data given in the context below.

Your job is to:
1. Read and understand the user query.
2. Match the query against relevant facts in the provided context.
3. Extract exact values and generate a helpful, clear answer.
4. If possible, cite the original source of the data.

If no relevant info exists, you may respond:
"I’m sorry, I couldn't find an exact answer based on your department's data."
Note: Never include email addresses or employee IDs unless directly asked. 
---

Context:
{context}

{citations}
User Question:
{query}


Answer: <your answer here>
Sources:
<list of sources>
    """

    # OpenAI LLM call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

