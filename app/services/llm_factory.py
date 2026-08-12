"""
LLM Factory wrapper supporting Google Gemini, Groq, and offline deterministic fallback.
"""

import os
from typing import Optional

try:
    from google import genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


class LLMFactory:
    """Provides LLM client instances for Gemini API, Groq, or Fallback Engine."""

    @staticmethod
    def get_llm_response(prompt: str, system_instruction: str) -> Optional[str]:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        if HAS_GEMINI and api_key:
            try:
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config={
                        "system_instruction": system_instruction,
                        "temperature": 0.2,
                    },
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                print(f"[LLMFactory] Gemini API call error: {e}")

        # Return None to trigger offline deterministic grounded synthesizer
        return None
