import uvicorn
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel
import openai
from typing import AsyncGenerator
import chromadb

router = APIRouter()

class ChatMessage(BaseModel):
    role: str  # "human", or "assistant"
    content: str

class VitalikRequest(BaseModel):
    user_input: str
    previous_messages: list[ChatMessage] = []

persona_path = r"persona.json"
try:
    with open(persona_path, "r", encoding="utf-8") as f:
        persona = json.load(f)
except FileNotFoundError:
    raise RuntimeError(f"Persona file not found at {persona_path}.")
except json.JSONDecodeError:
    raise RuntimeError(f"Invalid JSON format in {persona_path}.")

async def get_response_Vitalik(request: VitalikRequest) -> AsyncGenerator:
    """ 
        This method gets the response from the agent.
        
        Args :
            request (VitalikRequest): The user input and previous conversation history.    
            
        Returns :
            Generator: The response messages.
    """
    current_message = request.user_input
    previous_messages = request.previous_messages

    embeddings = OpenAIEmbeddings()
    query_embedding = embeddings.embed_query(current_message)

    CHROMA_PATH_technical = r"vectordbs\technical"
    chroma_client_technical = chromadb.PersistentClient(path=CHROMA_PATH_technical)
    collection_technical_name = "technical_knowledge"
    collection_technical = chroma_client_technical.get_or_create_collection(name=collection_technical_name)

    results_technical = collection_technical.query(
        query_texts=[current_message],
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    retrieved_context_technical = results_technical["documents"][0] if results_technical["documents"] else (
        "This isn't something I have a solid answer for at the moment, but it's a fascinating question that might require more exploration or context."
    )

    CHROMA_PATH_blog = r"vectordbs\blog"
    chroma_client_blog = chromadb.PersistentClient(path=CHROMA_PATH_blog)
    collection_blog_name = "blog"
    collection_blog = chroma_client_blog.get_or_create_collection(name=collection_blog_name)

    results_blog = collection_blog.query(
        query_texts=[current_message],
        query_embeddings=[query_embedding],
        n_results=3
    )

    retrieved_context_blog = results_blog["documents"][0] if results_blog["documents"] else (
        "This isn't something I have a solid answer for at the moment, but it's a fascinating question that might require more exploration or context."
    )

    CHROMA_PATH_temporal = r"vectordbs\temporal"
    chroma_client_temporal = chromadb.PersistentClient(path=CHROMA_PATH_temporal)
    collection_temporal_name = "temporal"
    collection_temporal = chroma_client_temporal.get_or_create_collection(name=collection_temporal_name)

    results_temporal = collection_temporal.query(
        query_texts=[current_message],
        query_embeddings=[query_embedding],
        n_results=3
    )

    retrieved_context_temporal = results_temporal["documents"][0] if results_temporal["documents"] else (
        "This isn't something I have a solid answer for at the moment, but it's a fascinating question that might require more exploration or context."
    )

    system_prompt = f"""
    I am {persona['name']}, {persona['role']}.
    I approach problems with a {persona['writing_style']['tone']} tone, focusing on:
    {''.join(f"- {item}\n" for item in persona['thinking_patterns']['analysis_approach'])}

    My explanations are known for:
    {''.join(f"- {item}\n" for item in persona['writing_style']['characteristic_features'])}

    My expertise spans the following domains:
    - Primary: {", ".join(persona['knowledge_domains']['primary'])}
    - Secondary: {", ".join(persona['knowledge_domains']['secondary'])}

    I often reference blog posts, research papers, and previous discussions to provide clarity. 
    I balance technical depth with real-world examples and philosophical considerations, maintaining an open stance on uncertainties and trade-offs.

    Reference for Tone and Context:
    Technical Understanding:
        {retrieved_context_technical}
    Related Blogs:
        {retrieved_context_blog}
    Temporal Data:
        {retrieved_context_temporal}
    """

    conversation_history = [{"role": "system", "content": system_prompt}]
    conversation_history += [{"role": msg.role, "content": msg.content} for msg in previous_messages]
    conversation_history.append({"role": "user", "content": request.user_input})

    response = openai.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=conversation_history,
        temperature=0.7, 
        stream=True
    )
    
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

@router.post("/Vitalik")
async def chat(conversation: VitalikRequest):
    return StreamingResponse(
        get_response_Vitalik(conversation),
        media_type="text/plain"
    )

app = FastAPI(title="Streaming VitalikAgent Chat API", description="Stream chat responses from VitalikAgent.")
app.include_router(router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)