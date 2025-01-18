import json
import asyncio
from string import Formatter
from typing import List, Dict, Any
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
import chromadb
from langchain.callbacks.base import BaseCallbackHandler
from models import ReasoningLayer
from utils import VitalikUtils

class VitalikAgent:
    def __init__(self, persona_path: str = 'persona.json'):
        """Initialize the VitalikAgent with persona and necessary resources"""
        print(f"Current working directory: {os.getcwd()}")
        print(f"Looking for persona file at: {os.path.abspath(persona_path)}")

        # Load persona configuration
        self.persona = self._load_persona(persona_path)

        # Initialize LLM
        self.llm = self._initialize_llm()

        # Initialize Utils
        self.utils = VitalikUtils(self.llm)

        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings()

        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            return_messages=True
        )

        # Initialize vector databases
        self.tech_collection, self.blog_collection, self.temporal_collection = self._initialize_vector_dbs()

        # Initialize tools and agent
        self.tools = self._create_tools()
        self.agent = self._create_agent()

    def _load_persona(self, persona_path: str) -> Dict[str, Any]:
        """Load persona data from a JSON file or use default values if unavailable"""
        try:
            if not os.path.exists(persona_path):
                print(f"Persona file not found: {persona_path}. Using default configuration.")
                return {"name": "Vitalik", "role": "Blockchain Visionary", "system_prompt_template": ""}

            with open(persona_path, 'r', encoding='utf-8') as f:
                persona = json.load(f)

            required_fields = ['name', 'role', 'system_prompt_template']
            missing_fields = [field for field in required_fields if field not in persona]
            if missing_fields:
                raise ValueError(f"Missing required fields in persona.json: {missing_fields}")

            return persona

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {persona_path}. Error: {str(e)}")
        except Exception as e:
            raise Exception(f"Error loading persona file: {str(e)}")

    def _initialize_llm(self):
        """Initialize the LLM with streaming"""
        try:
            return ChatOpenAI(
                model="gpt-4-turbo-preview",
                temperature=0.7,
                streaming=True
            )
        except Exception as e:
            raise Exception(f"Error initializing ChatOpenAI: {str(e)}")

    def _initialize_vector_dbs(self):
        """Initialize vector databases and collections"""
        try:
            tech_db = chromadb.PersistentClient(path="./vectordbs/technical")
            blog_db = chromadb.PersistentClient(path="./vectordbs/blog")
            temporal_db = chromadb.PersistentClient(path="./vectordbs/temporal")

            return (
                tech_db.get_collection("technical_knowledge"),
                blog_db.get_collection("blog_knowledge"),
                temporal_db.get_collection("temporal_knowledge")
            )
        except Exception as e:
            print(f"Error initializing vector databases: {str(e)}")
            return None, None, None

    def _create_tools(self) -> List[Tool]:
        """Create tools for the agent"""
        return [
            Tool(
                name="search_technical_knowledge",
                description="Search through technical papers and documentation about Ethereum",
                func=self._search_technical_db,
                coroutine=self._search_technical_db
            ),
            Tool(
                name="search_blog_knowledge",
                description="Search through my blog posts and articles",
                func=self._search_blog_db,
                coroutine=self._search_blog_db
            ),
            Tool(
                name="search_temporal_knowledge",
                description="Search through my tweets and historical content",
                func=self._search_temporal_db,
                coroutine=self._search_temporal_db
            )
        ]

    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent"""
        try:
            simple_prompt = f"""I am {self.persona['name']}, {self.persona['role']}.
            I approach problems from first principles, combining technical depth with philosophical considerations."""

            prompt = ChatPromptTemplate.from_messages([
                ("system", simple_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])

            agent = create_openai_functions_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=prompt
            )

            return AgentExecutor(
                agent=agent,
                tools=self.tools,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True
            )
        except Exception as e:
            raise Exception(f"Error creating agent: {str(e)}")

    async def _search_technical_db(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search the technical knowledge base"""
        if not self.tech_collection:
            print("Technical knowledge base unavailable.")
            return []

        try:
            query_embedding = self.embeddings.embed_query(query)
            results = self.tech_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            return self.utils.format_results(results, "technical")
        except Exception as e:
            print(f"Error in technical search: {str(e)}")
            return []

    async def _search_blog_db(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search the blog knowledge base"""
        if not self.blog_collection:
            print("Blog knowledge base unavailable.")
            return []

        try:
            query_embedding = self.embeddings.embed_query(query)
            results = self.blog_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            return self.utils.format_results(results, "blog")
        except Exception as e:
            print(f"Error in blog search: {str(e)}")
            return []

    async def _search_temporal_db(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search the temporal knowledge base"""
        if not self.temporal_collection:
            print("Temporal knowledge base unavailable.")
            return []

        try:
            query_embedding = self.embeddings.embed_query(query)
            results = self.temporal_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            return self.utils.format_results(results, "temporal")
        except Exception as e:
            print(f"Error in temporal search: {str(e)}")
            return []

    async def query(self, user_input: str, stream_handler: BaseCallbackHandler = None) -> Dict:
        """Process a user query through the agent"""
        try:
            callbacks = [stream_handler] if stream_handler else []
            agent_response = await self.agent.ainvoke(
                {"input": user_input},
                config={"callbacks": callbacks}
            )

            return {"response": agent_response['output']}
        except Exception as e:
            print(f"Error during query processing: {str(e)}")
            return {"error": str(e)}
