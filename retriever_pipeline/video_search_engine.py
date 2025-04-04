import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# if needed for extended functionality
from video_processor import VideoProcessor
from document_manager import DocumentManager
from config import Config

# Initialize GenAI client (replace with your actual API key)
from google import genai
from google.genai import types
client = genai.Client(api_key='xxxxxxxxxxxxxxxxxx')


class VideoSearchEngine:
    """Handles video content search and response generation"""

    def __init__(self):
        self.vector_store = DocumentManager.load_vector_store()
        self.cross_encoder = CrossEncoder(Config.CROSS_ENCODER_MODEL)
        self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)

    def search(self, query: str) -> dict:
        """Main search interface"""
        # Retrieve candidate documents for better re-ranking.
        faiss_results = self.vector_store.similarity_search_with_score(
            query, k=20)
        if not faiss_results:
            return {
                "query": query,
                "answer": "No relevant document found.",
                "video_id": None,
                "top_timestamps": []
            }
        # Rerank using a cross-encoder with a stricter confidence threshold.
        reranked_results = self._rerank_results(query, faiss_results)
        if not reranked_results:
            return {
                "query": query,
                "answer": "No relevant document found.",
                "video_id": None,
                "top_timestamps": []
            }
        return self._generate_response(query, reranked_results)

    def _rerank_results(self, query: str, results: list) -> list:
        """Rerank results using cross-encoder and filter with a higher confidence threshold."""
        candidates = [doc.page_content for doc, _ in results]
        cross_input = [[query, doc] for doc in candidates]
        scores = self.cross_encoder.predict(cross_input)
        # Convert raw scores to probabilities using a sigmoid.
        probs = 1 / (1 + np.exp(-np.array(scores)))
        reranked = []
        for i, prob in enumerate(probs):
            if prob >= 0.7:
                reranked.append({
                    "content": candidates[i],
                    "confidence": float(prob),
                    "metadata": results[i][0].metadata
                })
        return reranked

    def _refine_with_gemini(self, query: str, result: dict) -> str:
        """Refine raw content into a concise answer using enhanced context."""
        prompt = f""" Query: {query}
                  Context: {result}
                """

        system_instruction = """You are an expert video metadata analyzer. When given a query along with context and video metadata (including timestamps), your task is to generate a detailed yet concise answer that captures all the nuances of the content and answers the query. Your response must strictly adhere to the following format:
<Provide your detailed answer here, clearly outlining each point or step as needed. Do not begin with any filler words or introductory phrases. If the answer corresponds to multiple metadata segments, merge them into a single timestamp range that starts with the earliest start time and ends with the latest end time of the response.>

Timestamps:
{<earliest_start_time>-<latest_end_time>: <Provide a consolidated caption summarizing the overall content of the answer>}

Ensure that:
1. The answer is directly responsive to the query and uses the context provided.
2. The final output includes only the Answer section and the Timestamps section exactly as shown.
3. If the answer spans multiple timestamp segments, consolidate the timestamps into one range covering the earliest start to the latest end.
4. No extra commentary, explanations, or filler text is included outside the specified format.

Follow these instructions exactly to produce the required output.
"""
        attempt = 3
        response = client.models.generate_content(
            model=Config.GENAI_MODEL,
            contents=[types.Content(
                role="user",
                parts=[types.Part(text=prompt)]
            )],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=Config.TEMPERATURE + (attempt * 0.1)
            )
        )
        print("Gemini response:", response)
        refined = response.text.strip()
        return refined

    def _generate_response(self, query: str, results: list) -> dict:
        """Generate final response with aggregated details."""
        top_result = results[0]
        video_id = top_result["metadata"]["video_id"]
        file_name = f"{Config.OUTPUT_DIR}/{video_id}_metadata.json"
        with open(file_name, 'r', encoding="utf-8") as file:
            metadata_contents = json.load(file)

        confidence = top_result["confidence"]
        print("Top result confidence:", confidence)
        refined_answer = self._refine_with_gemini(query, metadata_contents)
        return {
            "query": query,
            "answer": refined_answer,
            "video_id": video_id,
            "confidence": confidence
        }
