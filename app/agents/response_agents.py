import json
import re
import logging
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from ..utils.config import GROQ_API_KEY, LLM_MODEL, GROQ_BASE_URL
logger = logging.getLogger(__name__)
class DraftGeneratorAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model=LLM_MODEL, openai_api_key=GROQ_API_KEY,
                              openai_api_base=GROQ_BASE_URL, temperature=0.2)
    def generate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("query", "")
        chunks = state.get("retrieved_chunks", [])
        rewrite_feedback = state.get("critic_feedback", "")
        context = "\n\n".join([
            f"[Source: {c.get('metadata', {}).get('source', 'N/A')}] {c['text']}"
            for c in chunks[:5]
        ]) if chunks else "No context available."
        if rewrite_feedback:
            prompt = ChatPromptTemplate.from_template(
                "Rewrite your response addressing this feedback:\n{feedback}\n\n"
                "Original question: {query}\n"
                "Context from federal regulations:\n{context}\n\n"
                "Provide a corrected response with proper citations."
            )
            response = (prompt | self.llm).invoke({
                "feedback": rewrite_feedback, "query": query, "context": context
            })
        else:
            prompt = ChatPromptTemplate.from_template(
                "You are an expert on US Federal Regulations.\n"
                "Answer this question using ONLY the provided context.\n\n"
                "Context:\n{context}\n\n"
                "Question: {query}\n\n"
                "Rules:\n"
                "- Cite specific regulation sources\n"
                "- If context doesn't contain the answer, say so\n"
                "- Be precise with dates and document numbers\n"
            )
            response = (prompt | self.llm).invoke({"context": context, "query": query})
        return {"draft": response.content, "status": "Draft generated"}
class CriticAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model=LLM_MODEL, openai_api_key=GROQ_API_KEY,
                              openai_api_base=GROQ_BASE_URL, temperature=0)
    def review(self, state: Dict[str, Any]) -> Dict[str, Any]:
        draft = state.get("draft", "")
        chunks = state.get("retrieved_chunks", [])
        context = "\n".join([c["text"] for c in chunks[:5]])
        prompt = ChatPromptTemplate.from_template(
            "You are a fact-checker for regulatory content.\n"
            "Review this response against the source context.\n\n"
            "Response:\n{draft}\n\n"
            "Source context:\n{context}\n\n"
            "Check:\n"
            "1. Are all claims supported by the context?\n"
            "2. Are citations present and correct?\n"
            "3. Is any information hallucinated?\n\n"
            "Return JSON: {{\"score\": 1-10, \"feedback\": \"...\", \"hallucinations\": []}}\n"
            "Return ONLY JSON."
        )
        response = (prompt | self.llm).invoke({"draft": draft, "context": context})
        try:
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            report = json.loads(content)
            score = int(report.get("score", 7))
        except (json.JSONDecodeError, ValueError):
            numbers = re.findall(r'\b([1-9]|10)\b', response.content)
            score = int(numbers[0]) if numbers else 7
            report = {"score": score, "feedback": response.content, "hallucinations": []}
        iterations = state.get("critic_iterations", 0) + 1
        logger.info(f"Critic: score={score}/10 (iteration {iterations})")
        return {
            "critic_score": score,
            "critic_feedback": report.get("feedback", ""),
            "critic_hallucinations": report.get("hallucinations", []),
            "critic_iterations": iterations,
            "status": f"Critic score: {score}/10"
        }
class ValidatorAgent:
    def validate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        draft = state.get("draft", "")
        issues = []
        if len(draft) < 50:
            issues.append("Response too short")
        if len(draft) > 5000:
            issues.append("Response too long — consider summarizing")
        is_valid = len(issues) == 0
        logger.info(f"Validator: {'PASSED' if is_valid else 'FAILED'} ({len(issues)} issues)")
        return {
            "is_valid": is_valid,
            "validation_issues": issues,
            "final_response": draft,
            "status": "Validation complete"
        }
