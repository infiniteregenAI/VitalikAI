import json
from typing import List, Dict
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
import chromadb
from api_utils import VitalikUtils

class VitalikAgent:
    def __init__(self, persona_path: str = 'persona.json'):
        print(f"\n\tCurrent working directory: {os.getcwd()}")
        print(f"\n\tLooking for persona file at: {os.path.abspath(persona_path)}")
        
        # Load persona configuration
        try:
            with open(persona_path, 'r', encoding='utf-8') as f:
                self.persona = json.load(f)
                print(f"\n\tSuccessfully loaded persona data: {self.persona}")
                
                # Validate required fields
                required_fields = ['name', 'role', 'system_prompt_template']
                missing_fields = [field for field in required_fields if field not in self.persona]
                if missing_fields:
                    raise ValueError(f"Missing required fields in persona.json: {missing_fields}")
                
        except FileNotFoundError:
            raise FileNotFoundError(f"Persona file not found at {persona_path}. Please ensure persona.json exists in the current directory.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {persona_path}. Error: {str(e)}")
        except Exception as e:
            raise Exception(f"Error loading persona file: {str(e)}")
        
        # Initialize LLM
        try:
            self.llm = ChatOpenAI(
                model="gpt-4-turbo-preview",
                temperature=0.7,
                stream=True
            )
        except Exception as e:
            print(f"Error initializing ChatOpenAI: {str(e)}")
            raise

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
        
        # Initialize vector DB clients
        try:
            self.tech_db = chromadb.PersistentClient(path="./vectordbs/technical")
            self.blog_db = chromadb.PersistentClient(path="./vectordbs/blog")
            self.temporal_db = chromadb.PersistentClient(path="./vectordbs/temporal")
            
            # Get collections
            self.tech_collection = self.tech_db.get_collection("technical_knowledge")
            self.blog_collection = self.blog_db.get_collection("blog_knowledge")
            self.temporal_collection = self.temporal_db.get_collection("temporal_knowledge")
        except Exception as e:
            print(f"Error initializing vector databases: {str(e)}")
            raise
        
        # Initialize tools and agent
        try:
            self.tools = self._create_tools()
            self.agent = self._create_agent()
            print("Successfully initialized agent!")
        except Exception as e:
            print(f"Error creating tools or agent: {str(e)}")
            raise

    async def _search_technical_db(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search technical knowledge base"""
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
        """Search blog knowledge base"""
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
        """Search temporal knowledge base"""
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

    def _create_tools(self) -> List[Tool]:
        """Create tools for the agent to use"""
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
        """Create the agent with custom prompt"""
        try:
            # Create a simplified system prompt for initial testing
            simple_prompt = f"""I am {self.persona['name']}, {self.persona['role']}. 
            I approach problems from first principles, combining technical depth with philosophical considerations."""
            
            # Use simplified prompt for now
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
            print(f"Error in _create_agent: {str(e)}")
            raise

    async def query(self, user_input: str) -> Dict:
        """Process a user query through the agent"""
        try:
            # Get agent's response with existing chat history
            agent_response = await self.agent.ainvoke({
                "input": user_input
            })
            
            # Ensure to handle the response correctly
            if isinstance(agent_response, dict) and 'output' in agent_response:
                response_output = agent_response['output']
            else:
                response_output = "Unexpected response format."
            
            return {"response": response_output}
        except Exception as e:
            print(f"Error during query processing: {str(e)}")
            return {
                "error": str(e),
                "status": "failed"
            }

    async def stream_query(self, user_input: str):
        """Stream the agent's response chunk by chunk."""
        try:
            agent_response = await self.agent.ainvoke({"input": user_input})
            if hasattr(agent_response, "stream"):
                async for chunk in agent_response.stream():
                    yield chunk  # Yield each chunk as it's received
            else:
                yield agent_response.get("output", "No output available")
        except Exception as e:
            yield f"Error: {str(e)}"        