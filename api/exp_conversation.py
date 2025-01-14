from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from fastapi import BackgroundTasks
import traceback
import json
import os
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
    CHROMA_PATH = r"vectordbs\technical"
    # CHROMA_PATH = r"vectordbs\Vitalik_db"

    if not os.path.exists(CHROMA_PATH):
        os.makedirs(CHROMA_PATH)
    os.chmod(CHROMA_PATH, 0o777)

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection_name = "technical_knowledge"
    # collection_name = "ai_persona"
    collection = chroma_client.get_or_create_collection(name=collection_name)
    current_message = request.current_message
    previous_messages = request.previous_messages

    results = collection.query(
        query_texts=[current_message],
        n_results=4
    )
    
    retrieved_context = results["documents"][0] if results["documents"] else (
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
    {retrieved_context}"""

    conversation_history = [{"role": "system", "content": system_prompt}]
    conversation_history += [{"role": "user", "content": msg} for msg in previous_messages]
    conversation_history.append({"role": "user", "content": current_message})

    response =  openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation_history,
        temperature=0.9,
        max_tokens=800,
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

# Create the FastAPI app and include the router
app = FastAPI()
app.include_router(router)
