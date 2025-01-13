from typing import List, Dict
from langchain_openai import ChatOpenAI

class VitalikUtils:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def format_results(self, results: Dict, db_type: str) -> List[Dict]:
        """Format search results with metadata"""
        return [{
            "content": doc,
            "metadata": meta,
            "type": db_type,
            "relevance_score": score
        } for doc, meta, score in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )]

    async def synthesize_thoughts(self, results: List[Dict]) -> str:
        """Synthesize thoughts based on search results"""
        prompt = f"""Given this information, what would I (Vitalik) think about this?
        Consider my writing style and typical approach to such topics.
        
        Information:
        {results}"""
        
        response = await self.llm.invoke(prompt)  # Assuming invoke is async
        print(f"Response type: {type(response)}")  # Debugging line
        
        # Check if the response is a stream
        if hasattr(response, 'stream'):
            # Collect the streamed output
            output = ""
            async for chunk in response.stream():
                output += chunk
            return output
        else:
            return response  # Handle non-streaming response

    async def draw_conclusion(self, results: List[Dict]) -> str:
        """Draw conclusions from search results"""
        prompt = f"""Based on these findings, what concrete insights would I (Vitalik) focus on?
        Frame it in my characteristic style of combining technical and philosophical perspectives.
        
        Findings:
        {results}"""
        
        response = await self.llm.invoke(prompt)  # Assuming invoke is async
        
        # Check if the response is a stream
        if hasattr(response, 'stream'):
            # Collect the streamed output
            output = ""
            async for chunk in response.stream():
                output += chunk
            return output
        else:
            return response  # Handle non-streaming response
    def calculate_confidence(self, results: List[Dict]) -> float:
        """Calculate confidence score based on result relevance"""
        if not results:
            return 0.0
        avg_relevance = sum(r["relevance_score"] for r in results) / len(results)
        return 1 - (avg_relevance / 2)