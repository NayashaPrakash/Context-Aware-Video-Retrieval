import json
import re
import time
import os
import json
from pathlib import Path
from typing import Dict, Any

from google import genai
from google.genai import types

from config import Config

client = genai.Client(api_key='xxxxxxxxxxxxxxxxxx')


class VideoProcessor:
    """Handles video upload and metadata extraction using Gemini"""

    PROMPTS = """
      Analyze the provided video content and generate a JSON object with the following structure:

      {
        "summary": "Based on the captions and descriptions and the video, generate a detailed summary.",
        "keywords": "Comma-separated list of 5-10 keywords from the video.",
        "timestamp_prompt": [
          {
            "start": "Start timestamp in HH:MM:SS format",
            "end": "End timestamp in HH:MM:SS format",
            "caption": "Spoken content at this timestamp.",
            "description": "Visual elements at this timestamp."
          },
          ... (more timestamps as needed)
        ]
      }

      Instructions:

      1.  **summary:** Do not start with 'Here is a summary' or other extra content in beginning. Based on the captions and descriptions and the video, generate a detailed summary which captures all the context and plot if present in the video.
      2.  **keywords:** Extract 5-10 relevant keywords from the video and list them as a comma-separated string.
      3.  **timestamp_prompt:**
          * Identify key segments of the video.
          * For each segment, create a dictionary with the following keys:
              * **start:** The start timestamp of the segment in HH:MM:SS format.
              * **end:** The end timestamp of the segment in HH:MM:SS format.
              * **caption:** The spoken content during that segment.
              * **description:** A detailed description of the visual elements displayed during that segment.
          * Include multiple timestamp entries to cover the entire video.
      """

    @classmethod
    def process_videos(cls):
        """Main processing pipeline for video files"""
        video_files = [f for f in Config.VIDEO_DIR.iterdir(
        ) if f.suffix in ('.mp4', '.avi', '.mov')]

        for video_path in video_files:
            print(f"\nProcessing {video_path.name}...")
            try:
                file_upload = cls._upload_video(video_path)
                metadata = cls._generate_metadata(video_path.name, file_upload)
                cls._save_metadata(video_path, metadata)
            except Exception as e:
                print(f"Error processing {video_path.name}: {str(e)}")

    @classmethod
    def _upload_video(cls, video_path: Path) -> types.File:
        """Upload video to GenAI and wait for processing"""
        file_upload = client.files.upload(file=video_path)

        while file_upload.state == "PROCESSING":
            print("Waiting for video processing...")
            time.sleep(10)
            file_upload = client.files.get(name=file_upload.name)

        if file_upload.state != "ACTIVE":
            raise RuntimeError(f"Failed to process {video_path.name}")
        return file_upload

    @classmethod
    def _generate_metadata(cls, video_id: str, file_upload: types.File) -> Dict[str, Any]:
        """Generate metadata using Gemini model"""
        metadata = {
            "video_id": video_id,
            "filename": file_upload.name
        }

        response_text = cls._get_ai_response(
            file_upload=file_upload,
            prompt=cls.PROMPTS,
            system_instruction=f"You are a professional video analysis AI. Your task is to analyze any input video and generate detailed, accurate, and structured information about it based on the given ${cls.PROMPTS}."
        )

        text = response_text.strip()
        text = re.sub(r"^```(json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        try:
            json_output = json.loads(text)
            metadata.update(json_output)
            # Ensure filename is correct.
            metadata["filename"] = file_upload.name
        except json.JSONDecodeError:
            print("Error: Could not parse JSON response.")
            print(response_text)
            metadata["error"] = "Could not parse JSON response from AI."

        return metadata

    @classmethod
    def _get_ai_response(cls, file_upload: types.File, prompt: str, system_instruction: str) -> str:
        """Helper function for Gemini API calls with retry logic"""
        for attempt in range(Config.RETRY_ATTEMPTS):
            try:
                response = client.models.generate_content(
                    model=Config.GENAI_MODEL,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[types.Part.from_uri(
                                file_uri=file_upload.uri,
                                mime_type=file_upload.mime_type
                            )]
                        ),
                        prompt
                    ],
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=Config.TEMPERATURE + (attempt * 0.1)
                    )
                )
                return response.text.strip()
            except Exception as e:
                if attempt == Config.RETRY_ATTEMPTS - 1:
                    return f"Error: {str(e)}"
                time.sleep(2 ** attempt)
        return "Content unavailable"

    @classmethod
    def _save_metadata(cls, video_path: Path, metadata: Dict[str, Any]):
        """Save metadata to JSON file"""
        output_path = Config.OUTPUT_DIR / \
            f"{metadata['video_id']}_metadata.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        print(f"Saved metadata to {output_path.name}")
