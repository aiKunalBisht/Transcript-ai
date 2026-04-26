# rag_retriever.py
# RAG (Retrieval-Augmented Generation) pipeline
#
# Flow:
#   1. User asks a question about their meetings
#   2. Embed the question using sentence-transformers
#   3. Retrieve top-K relevant meeting passages from ChromaDB
#   4. Pass retrieved context + question to LLM
#   5. LLM answers grounded in actual meeting data
#
# This prevents hallucination in Q&A — LLM only uses
# retrieved meeting content, not its training data.

import os

try:
    from rags.meeting_store import search_meetings, CHROMADB_AVAILABLE
except ImportError:
    CHROMADB_AVAILABLE = False
    def search_meetings(*args, **kwargs): return []


def build_rag_prompt(question: str, retrieved_meetings: list) -> str:
    """Build prompt with retrieved meeting context."""
    if not retrieved_meetings:
        return f"""You are a Japanese business meeting analyst.
The user asked: {question}

No relevant past meetings found in the database.
Tell the user to analyze some meetings first, then you can answer questions about them."""

    context_blocks = []
    for i, m in enumerate(retrieved_meetings, 1):
        date     = m.get("date", "")[:10]
        lang     = m.get("language", "unknown")
        risk     = m.get("soft_risk", "NONE")
        keigo    = m.get("keigo_level", "unknown")
        excerpt  = m.get("excerpt", "")
        sim      = m.get("similarity", 0)
        context_blocks.append(
            f"Meeting {i} (date:{date}, lang:{lang}, soft_rejection:{risk}, keigo:{keigo}, relevance:{sim:.0%}):\n{excerpt}"
        )

    context = "\n\n---\n\n".join(context_blocks)

    return f"""You are a Japanese business meeting analyst.
Answer the user's question using ONLY the meeting excerpts provided below.
Do not use information outside these excerpts.
If the answer is not in the excerpts, say so clearly.

RETRIEVED MEETING CONTEXT:
{context}

USER QUESTION: {question}

Answer concisely and cite which meeting number your answer comes from."""


def ask_about_meetings(
    question: str,
    n_context: int = 3,
    filter_language: str = None,
    filter_risk: str = None,
) -> dict:
    """
    RAG pipeline: retrieve relevant meetings → answer question.

    Returns:
        answer:    LLM response grounded in meeting data
        sources:   List of meetings used as context
        method:    'rag_langchain' | 'rag_direct' | 'no_data'
    """
    if not CHROMADB_AVAILABLE:
        return {
            "answer": "Meeting storage not available. Install chromadb: pip install chromadb",
            "sources": [],
            "method": "unavailable"
        }

    # Step 1: Retrieve relevant meetings
    retrieved = search_meetings(
        query=question,
        n_results=n_context,
        filter_language=filter_language,
        filter_risk=filter_risk,
    )

    if not retrieved:
        return {
            "answer": "No meetings found in the database yet. Analyze some meetings first.",
            "sources": [],
            "method": "no_data"
        }

    # Step 2: Build RAG prompt
    prompt = build_rag_prompt(question, retrieved)

    # Step 3: Call LLM with retrieved context
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass

    answer = None
    method = "rag_direct"

    # Try LangChain first
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage
        from langchain_core.output_parsers import StrOutputParser

        llm    = ChatGroq(api_key=api_key, model="llama-3.1-8b-instant",
                          temperature=0.2, max_tokens=800)
        chain  = llm | StrOutputParser()
        answer = chain.invoke([HumanMessage(content=prompt)])
        method = "rag_langchain"
    except Exception:
        pass

    # Fallback to direct Groq
    if not answer:
        try:
            import requests
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "llama-3.1-8b-instant",
                      "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0.2, "max_tokens": 800},
                timeout=30
            )
            r.raise_for_status()
            answer = r.json()["choices"][0]["message"]["content"]
            method = "rag_direct"
        except Exception as e:
            answer = f"Could not generate answer: {str(e)[:80]}"
            method = "error"

    return {
        "answer":  answer,
        "sources": retrieved,
        "method":  method,
        "context_meetings": len(retrieved),
    }


if __name__ == "__main__":
    import json

    # Test RAG pipeline
    result = ask_about_meetings(
        question="Which meetings had soft rejection signals?",
        n_context=3
    )
    print(f"Method: {result['method']}")
    print(f"Sources: {result['context_meetings']} meetings")
    print(f"\nAnswer:\n{result['answer']}")