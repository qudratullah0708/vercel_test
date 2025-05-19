from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import Optional, List, Dict, Any
import json
from datetime import datetime
import re

from tavily import TavilyClient
from groq import Groq
from dotenv import load_dotenv

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://lead-genius-suite.vercel.app", "http://http://localhost:8080"],  # Replace with actual domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load .env
load_dotenv()
model_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Clients
tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None
groq_client = Groq(api_key=model_api_key) if model_api_key else None


def Scrap_News(topic: str) -> str:
    if not tavily_client:
        raise ValueError("Tavily API key is missing.")

    print(f"🔍 Fetching News for: {topic}")
    response = tavily_client.search(topic, search_depth="advanced")

    if not response or "results" not in response:
        raise ValueError("Invalid response from Tavily API.")

    content = "\n".join(
        [
            f"Source: {r.get('url', 'Unknown')}\nTitle: {r.get('title', 'Unknown')}\nContent: {r.get('content', 'No content')}\n---"
            for r in response.get("results", [])
            if "content" in r
        ]
    )
    print("✅ News content fetched.")
    return content


def ExtractContent(content: str, query: str) -> List[Dict[str, Any]]:
    if not groq_client:
        raise ValueError("Groq API key is missing.")

    prompt = f"""
You are an expert lead generation AI that extracts business contact information from web content. Your task is to extract relevant leads from the provided content based on the search query: "{query}".

Input Content:
{content}

Instructions:

1. Extract Leads: Identify individuals in marketing, sales, or leadership roles at companies.
2. Required Fields: Each lead must include:
   - name
   - title
   - company
   - email (only if valid or inferred, no guessing)
   - phone (optional)
   - source
   - location

3. Include leads with at least two of the following: name, title, company.
4. Output quality leads only (preferably at least 5).
5. Do not hallucinate. Leave missing fields empty.
6. Output only a JSON array of lead objects.

Example Output:

[
  {{
    "name": "John Smith",
    "title": "Chief Marketing Officer",
    "company": "AI Solutions Inc",
    "email": "john.smith@aisolutions.com",
    "phone": "",
    "source": "LinkedIn",
    "location": "San Francisco, USA"
  }}
]
"""

    print("🧠 Extracting leads using Groq...")

    completion = groq_client.chat.completions.create(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        messages=[{"role": "user", "content": prompt}]
    )

    response = completion.choices[0].message.content.strip()

    try:
        json_match = re.search(r'\[[\s\S]*\]', response, re.DOTALL)
        json_str = json_match.group(0) if json_match else response
        leads = json.loads(json_str)

        if not leads or len(leads) < 1:
            raise ValueError("No leads extracted")

        print("✅ Leads extracted successfully.")
        return leads

    except Exception as e:
        print("❌ Failed to extract leads:", str(e))
        print("Raw LLM response:", response)
        return []  # Avoid crash


@app.get("/")
async def root():
    return {"message": "LeadGen API is running"}


@app.get("/api/search")
async def search(query: str):
    if not query or query.strip() == "":
        raise HTTPException(status_code=400, detail="Search query cannot be empty")

    try:
        if not tavily_api_key or not model_api_key:
            raise ValueError("Missing API keys")

        news_content = Scrap_News(query)
        leads = ExtractContent(news_content, query)

        if isinstance(leads, list):
            for i, lead in enumerate(leads):
                lead["id"] = str(i + 1)

            return {
                "success": True,
                "query": query,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "results": leads
            }

        raise HTTPException(status_code=500, detail="Unexpected response format")

    except Exception as e:
        print("❌ Error in /api/search:", str(e))
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# Local run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
