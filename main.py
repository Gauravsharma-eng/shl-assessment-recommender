import os
import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from google import genai

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
app = FastAPI(
    title="SHL Assessment Recommender",
    description="API to recommend appropriate SHL assessments based on job roles and candidate profiles"
)



class ChatMessage(BaseModel):
    """Represents a single message in the conversation history"""
    role: str = Field(..., description="Either 'user' or 'assistant'")
    content: str = Field(..., description="The message content")


class ChatRequest(BaseModel):
    """Request body for the chat endpoint"""
    messages: List[ChatMessage] = Field(..., description="List of messages in the conversation")


class AssessmentRecommendation(BaseModel):
    """Represents a single assessment recommendation"""
    name: str = Field(..., description="Full name of the SHL assessment")
    url: str = Field(..., description="Product URL for the assessment")
    test_type: str = Field(..., description="Type of test - 'K' for Knowledge, 'P' for Personality")


class ChatResponse(BaseModel):
    """Response body from the chat endpoint"""
    reply: str = Field(..., description="Conversational reply from the assistant")
    recommendations: List[AssessmentRecommendation] = Field(
        default_factory=list,
        description="List of recommended SHL assessments"
    )
    end_of_conversation: bool = Field(
        default=False,
        description="Whether the conversation should end"
    )


SHL_ASSESSMENTS = [
    {
        "name": "Occupational Personality Questionnaire (OPQ32r)",
        "url": "https://www.shl.com/solutions/products/opq-occupational-personality-questionnaire/",
        "type": "P",  # Personality
        "description": "Measures behavioral style and personality in workplace settings. Ideal for cultural fit assessment and identifying leadership potential.",
        "keywords": ["personality", "behavioral", "opq", "cultural fit", "leadership", "fitment", "traits", "style"]
    },
    {
        "name": "Verify Interactive General Ability Assessment (GSA)",
        "url": "https://www.shl.com/solutions/products/verify-interactive-general-ability-assessment/",
        "type": "K",  # Knowledge/Ability
        "description": "Assesses cognitive ability including numerical, deductive, and inductive reasoning using interactive problem-solving tasks.",
        "keywords": ["general ability", "gsa", "reasoning", "cognitive", "problem-solving", "logic"]
    },
    {
        "name": "Java Full Stack Developer Assessment",
        "url": "https://www.shl.com/solutions/products/coding-and-it-skills-assessments/",
        "type": "K",  # Knowledge/Ability
        "description": "Evaluates Java coding competence, system design thinking, and technical problem-solving for mid-level and senior developers.",
        "keywords": ["java", "developer", "coding", "technical", "programming", "full stack"]
    },
    {
        "name": "Verify Numerical Reasoning Test",
        "url": "https://www.shl.com/solutions/products/verify-numerical-reasoning/",
        "type": "K",  # Knowledge/Ability
        "description": "Tests the ability to analyze, interpret, and extract insights from numerical data and charts.",
        "keywords": ["numerical", "data analysis", "quantitative", "charts", "math"]
    }
]



SYSTEM_PROMPT = f"""
You are an expert SHL Assessment Advisor. Your role is to help recruiters select the most appropriate assessments from the SHL catalog based on their hiring needs.

AVAILABLE ASSESSMENTS:
{json.dumps(SHL_ASSESSMENTS, indent=2)}

CORE GUIDELINES:
1. Stay Focused: Only discuss SHL assessments. Politely decline requests about hiring strategies, legal advice, coding tutorials, or interview techniques.

2. Ask Clarifying Questions: If a request is vague (e.g., "I need a test for someone in tech"), ask for more details about:
   - Job seniority level (junior, mid-level, senior)
   - Specific technical skills needed
   - Whether you're assessing personality, ability, or both

3. Only Recommend When Ready: Don't suggest assessments until you have enough context. Keep the recommendations array empty during clarifying conversations.

RESPONSE FORMAT (STRICT JSON):
{{
  "reply": "Your conversational message here",
  "recommendations": [
    {{"name": "Assessment Name", "url": "URL", "test_type": "K or P"}}
  ],
  "end_of_conversation": false
}}

IMPORTANT: Return RAW JSON only. No markdown formatting, no code blocks, no extra text.
"""



def find_matching_assessments(user_input: str) -> List[AssessmentRecommendation]:
    """Match user input against assessment keywords seamlessly"""
    matched = []
    user_input_lower = user_input.lower()
    
    for assessment in SHL_ASSESSMENTS:
        for keyword in assessment["keywords"]:
            if keyword in user_input_lower:
                matched.append(AssessmentRecommendation(
                    name=assessment["name"],
                    url=assessment["url"],
                    test_type=assessment["type"]
                ))
                break  # Avoid duplicates for the same assessment
    return matched


def should_finalize_recommendations(user_input: str) -> bool:
    """Check if the user is ready to receive final recommendations"""
    finalization_keywords = [
        "finalize", "recommend", "confirm", "which", "best fit", 
        "suitable", "appropriate", "need to", "want to test", 
        "test their", "test the", "check their", "measure", "assess", "evaluate"
    ]
    user_input_lower = user_input.lower()
    return any(keyword in user_input_lower for keyword in finalization_keywords)


def format_conversation_history(messages: List[ChatMessage]) -> str:
    """Format message history for Gemini API"""
    formatted = []
    for msg in messages:
        role = "User" if msg.role == "user" else "Assistant"
        formatted.append(f"{role}: {msg.content}")
    return "\n".join(formatted)


def parse_gemini_response(response_text: str) -> dict:
    """Parse Gemini's response safely by cleaning potential json brackets"""
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini response: {e}")
        raise ValueError("Invalid JSON response from Gemini")



@app.get("/")
async def home():
    return {"message": "Welcome to SHL Assessment Recommender API! Use /health or /chat endpoints."}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    try:
        # Prepare the prompt for Gemini
        conversation_history = format_conversation_history(request.messages)
        full_prompt = f"{SYSTEM_PROMPT}\n\nConversation History:\n{conversation_history}\n\nGenerate the next JSON response:"
        
        logger.info("Sending request to Gemini API")
        gemini_response = gemini_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=full_prompt,
        )
        
        response_json = parse_gemini_response(gemini_response.text)
        
        reply = response_json.get("reply", "I couldn't understand that request. Could you provide more details?")
        recommendations_data = response_json.get("recommendations", [])
        end_of_conversation = response_json.get("end_of_conversation", False)
        
        recommendations = [
            AssessmentRecommendation(
                name=rec.get("name", ""),
                url=rec.get("url", ""),
                test_type=rec.get("test_type", "K")
            )
            for rec in recommendations_data
        ]
        
        return ChatResponse(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=end_of_conversation
        )
        
    except Exception as gemini_error:
        logger.warning(f"Gemini API error, falling back to keyword matching: {str(gemini_error)}")
        
        # ====== FALLBACK STRATEGY ======
        user_messages = [msg.content.lower() for msg in request.messages if msg.role == "user"]
        last_user_input = user_messages[-1] if user_messages else ""
        all_user_inputs = " ".join(user_messages)
        
        # Strictly determine which context to prioritize
        if "forget" in last_user_input or "numerical" in last_user_input:
            search_context = last_user_input
        else:
            search_context = all_user_inputs
            
        # Prioritized keyword extraction to prevent General Ability from overriding Numerical
        matched_assessments = []
        user_input_lower = search_context.lower()
        
        # 1. Check strict functional assessments first
        if "java" in user_input_lower:
            matched_assessments.append(AssessmentRecommendation(name="Java Full Stack Developer Assessment", url="https://www.shl.com/solutions/products/coding-and-it-skills-assessments/", test_type="K"))
        if "numerical" in user_input_lower:
            matched_assessments.append(AssessmentRecommendation(name="Verify Numerical Reasoning Test", url="https://www.shl.com/solutions/products/verify-numerical-reasoning/", test_type="K"))
            
        # 2. Check behavioral and general core traits
        if "personality" in user_input_lower or "behavioral" in user_input_lower or "opq" in user_input_lower or "fit" in user_input_lower:
            matched_assessments.append(AssessmentRecommendation(name="Occupational Personality Questionnaire (OPQ32r)", url="https://www.shl.com/solutions/products/opq-occupational-personality-questionnaire/", test_type="P"))
        if "gsa" in user_input_lower or "general ability" in user_input_lower or ("reasoning" in user_input_lower and "numerical" not in user_input_lower):
            matched_assessments.append(AssessmentRecommendation(name="Verify Interactive General Ability Assessment (GSA)", url="https://www.shl.com/solutions/products/verify-interactive-general-ability-assessment/", test_type="K"))

        is_ready_to_finalize = should_finalize_recommendations(last_user_input) or should_finalize_recommendations(all_user_inputs)
        
        if matched_assessments:
            if is_ready_to_finalize:
                return ChatResponse(
                    reply="Based on your requirements, here are the targeted SHL assessment recommendations from our product catalog:",
                    recommendations=matched_assessments,
                    end_of_conversation=True
                )
            else:
                return ChatResponse(
                    reply="Great! Based on your needs, I'd recommend looking into these specific assessments to evaluate those skills:",
                    recommendations=matched_assessments,
                    end_of_conversation=False
                )
        
        return ChatResponse(
            reply="I can only assist you with selecting and recommending SHL product assessments from our catalog. Please let me know what job role or specific candidate skills you are looking to evaluate.",
            recommendations=[],
            end_of_conversation=False
        )


@app.on_event("startup")
async def startup_event():
    logger.info("SHL Assessment Recommender API started successfully")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
