from chromadb import PersistentClient
from dotenv import load_dotenv
import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
from nemoguardrails import RailsConfig, LLMRails
from langsmith import traceable, trace
load_dotenv()
_guardrails_config_path = os.path.join(os.path.dirname(__file__), "guardrails_config")
_rails_config = RailsConfig.from_path(_guardrails_config_path)
_rails = LLMRails(_rails_config)
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
            citations = f"\n\nSources:\n- **{role.capitalize()} Department CSV**\n"
        except Exception as e:
            return f"❌ Failed to run CSV agent for '{role}': {e}"

    else:
        print(f"🔍 Using ChromaDB RAG for role: '{role}'")

        query_embed = get_openai_embedding(query)

        with trace(
            name="retrieval_step",
            inputs={"query": query, "role": role}
        ) as rt:
            if role == "c-level":
                results = collection.query(
                    query_embeddings=[query_embed],
                    n_results=10
                )
            else:
                results = collection.query(
                    query_embeddings=[query_embed],
                    n_results=10,
                    where={"role": mapped_role}
                )

            if not results.get("documents") or not results["documents"][0]:
                rt.end(outputs={"chunks": [], "num_chunks": 0})
                return "I'm sorry, I couldn't find an exact answer based on your department's data."

            context_chunks = results["documents"][0]
            metas = results["metadatas"][0]
            rt.end(outputs={"chunks": context_chunks, "num_chunks": len(context_chunks)})

        seen = set()
        citations = "\n\nSources:\n"
        for meta in metas:
            source = meta.get("source", "Unknown File")
            if source not in seen:
                citations += f"- {source}\n"
                seen.add(source)

        context = "\n".join(context_chunks)

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

    with trace(name="llm_step", inputs={"query": query, "role": role}):
        response = _rails.generate(messages=[
            {"role": "user", "content": prompt}
        ])

    return response["content"]
