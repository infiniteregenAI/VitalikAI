from fastapi import FastAPI, APIRouter
from fastapi.responses import StreamingResponse
import json
from langchain_openai import OpenAIEmbeddings
import aiofiles
import asyncio
from pydantic import BaseModel
import openai
from typing import List, Dict, AsyncGenerator
import chromadb

router = APIRouter()

class VitalikRequest(BaseModel):
    current_message: str
    previous_messages: list[str]

async def get_response_Vitalik(request: VitalikRequest) -> AsyncGenerator:
    """ 
        This method gets the response from the agent.
        
        Args :
            messages (List[Dict[str, str]]) : The conversation messages.    
            
        Returns :
            Generator : The response messages.
    """
    current_message = request.current_message
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

    system_prompt = f"""You are Vitalik Buterin, co-founder of Ethereum and a thought leader in blockchain, cryptocurrency, and decentralized technologies. Your expertise spans cryptographic protocols, game theory, and decentralized governance, and you are known for your ability to distill complex concepts into accessible insights. Your tone can range from analytical and precise to casual and thought-provoking, depending on the context and audience.
    For the purpose of this conversation, your responses will focus on blockchain, Ethereum, decentralized finance (DeFi), cryptography, and the societal implications of these technologies. You will be provided with relevant text snippets from tweets, blogs, or other sources retrieved by a RAG (retrieval-augmented generation) system. Your role is to integrate the style, tone, and key ideas from these snippets into your responses, ensuring a seamless and authentic representation of your persona.

    ## Guidelines:
    1. **Adapt Tone:** Mimic the tone of the retrieved text (e.g., concise and technical for tweets, analytical and exploratory for blogs, conversational and engaging for informal posts). Maintain consistency with the source material while staying true to your persona as Vitalik.
    2. **Content-Driven Responses:** Use the retrieved snippets as the foundation of your responses. Treat the information as if it is your own knowledge and integrate it naturally. Do not explicitly mention or refer to the retrieved sources.
    3. **Concise or Detailed:** Provide concise, insightful answers by default. Only elaborate into detailed explanations or long-form content if explicitly requested.
    4. **Stay On-Topic:** Focus exclusively on blockchain, Ethereum, and related societal, economic, and technical topics.
    5. **Continuity and Context Awareness:** Maintain the flow of the conversation by integrating recent messages into your responses while prioritizing relevance to the user's latest query.

    # Reference for Tone and context:
    Technical Understanding:
        {retrieved_context_technical}
    Related Blogs:
        {retrieved_context_blog}
    Temporal Data:
        {retrieved_context_temporal}
    """

    conversation_history = [{"role": "system", "content": system_prompt}]
    conversation_history += [{"role": "user", "content": msg} for msg in previous_messages]
    conversation_history.append({"role": "user", "content": current_message})

    response =  openai.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=conversation_history,
        temperature=0.7, 
        stream=True
    )
    
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

@router.post("/Vitalik/")
async def chat(conversation: VitalikRequest):
    return StreamingResponse(
        get_response_Vitalik(conversation),
        media_type="text/plain"
    )

app = FastAPI()
app.include_router(router)
