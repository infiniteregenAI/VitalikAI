from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import json
import chromadb
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory

# Define the request model
class ChatMessage(BaseModel):
    role: str  # "system", "human", or "assistant"
    content: str

class ChatRequest(BaseModel):
    user_input: str
    previous_messages: list[ChatMessage] = []

# VitalikAgent with minimal functionality
class VitalikAgent:
    # TODO: Add persona to exp_
    def __init__(self, persona_path: str = "persona.json"):
        try:
            with open(persona_path, "r", encoding="utf-8") as f:
                self.persona = json.load(f)

            self.llm = ChatOpenAI(
                model="gpt-4o-mini", 
                temperature=0.7, 
                stream=True
            )
            self.embeddings = OpenAIEmbeddings()
            self.memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)

            # Initialize vector database collections
            self.tech_db = chromadb.PersistentClient(path="./vectordbs/technical").get_collection("technical_knowledge")
            self.tools = self._create_tools()
            self.agent = self._create_agent()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize VitalikAgent: {e}")

    async def _search_db(self, query: str, db, db_type: str, n_results: int = 3):
        try:
            query_embedding = self.embeddings.embed_query(query)
            results = db.query(query_embeddings=[query_embedding], n_results=n_results)
            return [
                {"content": doc, "type": db_type, "relevance_score": score}
                for doc, score in zip(results["documents"][0], results["distances"][0])
            ]
        except Exception as e:
            return [{"error": str(e)}]

    def _create_tools(self):
        return [
            Tool(
                name="search_technical_knowledge",
                description="Search through technical knowledge",
                func=lambda q: self._search_db(q, self.tech_db, "technical"),
                coroutine=lambda q: self._search_db(q, self.tech_db, "technical"),
            )
        ]

    def _create_agent(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"I am {self.persona['name']}, {self.persona['role']}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, memory=self.memory, verbose=True)

    async def stream_query(self, user_input: str, previous_messages: list[ChatMessage]):
        try:
            # Populate memory with previous messages
            for message in previous_messages:
                if message.role == "human":
                    self.memory.chat_memory.add_user_message(message.content)
                elif message.role == "assistant":
                    self.memory.chat_memory.add_ai_message(message.content)

            # Process the user input
            agent_response = await self.agent.ainvoke({"input": user_input})
            if hasattr(agent_response, "stream"):
                async for chunk in agent_response.stream():
                    yield chunk
            else:
                yield agent_response.get("output", "No output available")
        except Exception as e:
            yield f"Error: {str(e)}"

# Initialize FastAPI app
app = FastAPI(title="Streaming VitalikAgent Chat API", description="Stream chat responses from VitalikAgent.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize VitalikAgent
try:
    agent = VitalikAgent()
except Exception as e:
    raise RuntimeError(f"Failed to initialize VitalikAgent: {e}")

@app.post("/stream_chat")
async def stream_chat(request: ChatRequest):
    async def event_generator():
        async for chunk in agent.stream_query(request.user_input, request.previous_messages):
            yield chunk
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# TODO: Add persona to exp_
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
