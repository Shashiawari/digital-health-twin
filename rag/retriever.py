"""
RAG Retriever
==============
Retrieves relevant patient contexts from ChromaDB and generates
natural-language responses using a local transformer model or an
enhanced template engine that handles arbitrary free-text queries.
"""

import re
from pathlib import Path
from typing import List, Dict

PROJ_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJ_ROOT / "rag" / "chroma_db"

# Lazy-loaded singletons
_embed_model = None
_chroma_collection = None
_generator = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def get_collection():
    global _chroma_collection
    if _chroma_collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _chroma_collection = client.get_collection("patients")
    return _chroma_collection


def get_generator():
    """
    Load a small local text-generation model (Flan-T5).
    Falls back to enhanced template-based generation if unavailable.
    """
    global _generator
    if _generator is None:
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch

            model_name = "google/flan-t5-base"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model.eval()
            _generator = {"model": model, "tokenizer": tokenizer}
            print("LLM loaded: google/flan-t5-base")
        except Exception as e:
            print(f"Could not load LLM, using template generator: {e}")
            _generator = "template"
    return _generator


# ── Retrieval ──────────────────────────────────────────────────────────

def retrieve_context(query: str, top_k: int = 5) -> List[Dict]:
    """
    Retrieve the most relevant patient records for a given query.
    Returns list of dicts with 'document', 'metadata', 'distance'.
    """
    collection = get_collection()
    embed_model = get_embed_model()

    query_embedding = embed_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    contexts = []
    for i in range(len(results["ids"][0])):
        contexts.append({
            "patient_id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    return contexts


# ── Generation ─────────────────────────────────────────────────────────

def generate_response(query: str, contexts: List[Dict]) -> str:
    """
    Generate a response to a free-text query using the enhanced
    template engine backed by retrieved patient context.

    The template engine uses intent detection to classify the query and
    then generates a rich, data-driven response from cohort statistics.
    This produces more consistent, clinically-grounded answers than
    a small local LLM (Flan-T5-base) which tends to copy raw text.
    """
    return _template_response(query, contexts)



# ── Enhanced Template Engine ───────────────────────────────────────────

def _compute_cohort_stats(contexts: List[Dict]) -> Dict:
    """Pre-compute statistics from the retrieved patient cohort."""
    n = len(contexts)
    if n == 0:
        return {}

    n_high = sum(1 for c in contexts if c["metadata"].get("target") == 1)
    ages = [c["metadata"].get("age", 0) for c in contexts]
    bmis = [c["metadata"].get("bmi", 0) for c in contexts]
    stresses = [c["metadata"].get("stress_level", 0) for c in contexts]
    anxieties = [c["metadata"].get("anxiety_score", 0) for c in contexts]

    return {
        "n": n,
        "n_high": n_high,
        "n_low": n - n_high,
        "pct_high": n_high / n * 100,
        "avg_age": sum(ages) / n,
        "avg_bmi": sum(bmis) / n,
        "avg_stress": sum(stresses) / n,
        "avg_anxiety": sum(anxieties) / n,
        "min_age": min(ages),
        "max_age": max(ages),
    }


def _detect_intent(query: str) -> str:
    """
    Classify the user's free-text query into an intent category.
    Handles a wide range of natural language questions.
    """
    q = query.lower().strip()

    # Risk / disease / health risks
    if re.search(r"(risk|danger|disease|diagnos|predict|chance|likelihood|prognos)", q):
        return "risk"

    # Mental health / stress / anxiety / sleep
    if re.search(r"(mental|stress|anxi|sleep|depress|psych|mood|insomnia|burnout)", q):
        return "mental"

    # Treatment / recommendation / what should / advice
    if re.search(r"(treat|recommend|advice|suggest|should|prevent|manag|improv|help|cure|therap)", q):
        return "recommendation"

    # Comparison / similar / like / profile
    if re.search(r"(similar|compar|like|match|profile|cohort|population)", q):
        return "similar"

    # Cause / why / factor / reason / contribute
    if re.search(r"(cause|why|factor|reason|contribut|lead|trigger|what.*(caus|driv))", q):
        return "cause"

    # Statistics / how many / prevalence / common
    if re.search(r"(statistic|how many|prevalenc|common|frequent|percent|rate|average)", q):
        return "statistics"

    # Specific patient / this patient / individual
    if re.search(r"(this patient|my patient|the patient|individual|specific|person)", q):
        return "patient_specific"

    # Age / demographic queries
    if re.search(r"(\d+.year.old|age|young|old|elder|senior)", q):
        return "demographic"

    # Lifestyle / activity / exercise / diet / BMI / weight
    if re.search(r"(lifestyle|activit|exercise|diet|bmi|weight|obes|overweight|sedentary|fit)", q):
        return "lifestyle"

    # General / catch-all
    return "general"


def _template_response(query: str, contexts: List[Dict]) -> str:
    """
    Generate a rich, natural-language response to ANY free-text query
    by combining intent detection with cohort statistics from the
    retrieved patient records.
    """
    if not contexts:
        return "No relevant patient data found for your query."

    stats = _compute_cohort_stats(contexts)
    intent = _detect_intent(query)
    top = contexts[0]

    # Build risk-factor snippets from cohort averages
    risk_snippets = []
    if stats["avg_bmi"] > 28:
        risk_snippets.append(f"elevated BMI ({stats['avg_bmi']:.1f})")
    if stats["avg_stress"] > 6:
        risk_snippets.append(f"high stress levels ({stats['avg_stress']:.1f}/10)")
    if stats["avg_anxiety"] > 10:
        risk_snippets.append(f"significant anxiety ({stats['avg_anxiety']:.0f}/21)")
    if stats["avg_age"] > 55:
        risk_snippets.append(f"older age group (avg {stats['avg_age']:.0f} years)")
    risk_phrase = ", ".join(risk_snippets) if risk_snippets else "no single dominant factor"

    # ── Intent-specific responses ──────────────────────────────────

    if intent == "risk":
        return (
            f"Based on analysis of {stats['n']} similar patient profiles, "
            f"{stats['n_high']}/{stats['n']} ({stats['pct_high']:.0f}%) have diagnosed heart disease. "
            f"Key risk factors include: {risk_phrase}. "
            f"Average age in this cohort: {stats['avg_age']:.0f} years. "
            f"{'Given the high prevalence among similar patients, comprehensive cardiovascular screening is strongly recommended.' if stats['pct_high'] > 50 else 'While risk is present, proactive monitoring and lifestyle modifications can significantly reduce cardiovascular events.'}"
        )

    if intent == "mental":
        high_risk_stress = []
        low_risk_stress = []
        for c in contexts:
            s = c["metadata"].get("stress_level", 0)
            if c["metadata"].get("target") == 1:
                high_risk_stress.append(s)
            else:
                low_risk_stress.append(s)
        avg_hr_stress = sum(high_risk_stress) / len(high_risk_stress) if high_risk_stress else 0
        avg_lr_stress = sum(low_risk_stress) / len(low_risk_stress) if low_risk_stress else 0

        return (
            f"Mental health analysis across {stats['n']} retrieved patient profiles: "
            f"Average stress level: {stats['avg_stress']:.1f}/10. "
            f"Average anxiety score (GAD-7): {stats['avg_anxiety']:.0f}/21. "
            f"{'Among high-risk cardiac patients in this cohort, stress averages ' + f'{avg_hr_stress:.1f}/10 compared to {avg_lr_stress:.1f}/10 in low-risk patients. ' if high_risk_stress and low_risk_stress else ''}"
            f"Research demonstrates a strong bidirectional relationship between mental health "
            f"and cardiovascular risk -- elevated stress and anxiety can increase cardiac event "
            f"probability by 1.5-2x through inflammation and autonomic dysfunction. "
            f"Integrated mental-physical health monitoring is recommended."
        )

    if intent == "recommendation":
        recs = []
        if stats["avg_bmi"] > 25:
            recs.append("maintain a healthy weight through balanced diet and regular exercise")
        if stats["avg_stress"] > 5:
            recs.append("implement stress management techniques such as meditation, therapy, or structured relaxation")
        if stats["avg_anxiety"] > 8:
            recs.append("consult a mental health professional for anxiety assessment")
        if stats["pct_high"] > 50:
            recs.append("schedule regular cardiovascular screenings")
        recs.append("aim for 150+ minutes per week of moderate physical activity")
        recs.append("monitor blood pressure and cholesterol levels regularly")

        return (
            f"Based on patterns from {stats['n']} similar patient profiles "
            f"({stats['pct_high']:.0f}% diagnosed with heart disease), the following is recommended: "
            f"{'; '.join(recs)}. "
            f"Early intervention and lifestyle modifications can significantly reduce cardiovascular risk, "
            f"especially when addressing both physical and mental health factors together."
        )

    if intent == "cause":
        return (
            f"Analysis of {stats['n']} similar patient records reveals several contributing factors "
            f"to cardiovascular risk: {risk_phrase}. "
            f"In this cohort, {stats['pct_high']:.0f}% have diagnosed heart disease. "
            f"Heart disease is multifactorial -- traditional risk factors (cholesterol, blood pressure, age) "
            f"interact with lifestyle factors (physical activity, BMI) and mental health indicators "
            f"(stress, anxiety, sleep quality). The combination of physical and psychological stressors "
            f"creates compounding risk through chronic inflammation and hormonal dysregulation."
        )

    if intent == "similar":
        top_sim = 1 - top["distance"]
        return (
            f"The most similar patient in the knowledge base is {top['patient_id']} "
            f"(similarity: {top_sim:.1%}). {top['document']} "
            f"Across all {stats['n']} similar profiles: {stats['n_high']} high-risk, "
            f"{stats['n_low']} low-risk. Average age: {stats['avg_age']:.0f}, "
            f"average BMI: {stats['avg_bmi']:.1f}."
        )

    if intent == "statistics":
        return (
            f"Statistics from {stats['n']} retrieved patient profiles: "
            f"Heart disease prevalence: {stats['pct_high']:.0f}% ({stats['n_high']}/{stats['n']}). "
            f"Age range: {stats['min_age']:.0f}-{stats['max_age']:.0f} years (avg {stats['avg_age']:.0f}). "
            f"Average BMI: {stats['avg_bmi']:.1f}. "
            f"Average stress level: {stats['avg_stress']:.1f}/10. "
            f"Average anxiety score: {stats['avg_anxiety']:.0f}/21. "
            f"These statistics are derived from the most semantically similar patient records "
            f"in the database to your query."
        )

    if intent == "patient_specific":
        # Give a comprehensive analysis of the top match
        m = top["metadata"]
        risk_str = "HIGH RISK" if m.get("target") == 1 else "LOW RISK"
        return (
            f"Patient profile analysis ({top['patient_id']}, {risk_str}): "
            f"{top['document']} "
            f"Among {stats['n']} similar patients, {stats['pct_high']:.0f}% have heart disease. "
            f"Key observations: {'elevated' if m.get('stress_level', 0) > 6 else 'moderate'} stress "
            f"({m.get('stress_level', 'N/A')}/10), "
            f"{'concerning' if m.get('anxiety_score', 0) > 10 else 'manageable'} anxiety "
            f"(GAD-7: {m.get('anxiety_score', 'N/A')}/21), "
            f"BMI: {m.get('bmi', 'N/A')}."
        )

    if intent == "demographic":
        return (
            f"Demographic analysis of {stats['n']} similar patients: "
            f"Age range: {stats['min_age']:.0f}-{stats['max_age']:.0f} years "
            f"(average: {stats['avg_age']:.0f}). "
            f"Heart disease prevalence in this group: {stats['pct_high']:.0f}%. "
            f"Average BMI: {stats['avg_bmi']:.1f}. "
            f"Cardiovascular risk generally increases with age, compounded by lifestyle "
            f"factors such as stress ({stats['avg_stress']:.1f}/10) and reduced physical activity."
        )

    if intent == "lifestyle":
        return (
            f"Lifestyle factor analysis across {stats['n']} similar patients: "
            f"Average BMI: {stats['avg_bmi']:.1f} "
            f"({'above healthy range' if stats['avg_bmi'] > 25 else 'within healthy range'}). "
            f"Average stress level: {stats['avg_stress']:.1f}/10. "
            f"In this cohort, {stats['pct_high']:.0f}% have heart disease. "
            f"Sedentary lifestyles and elevated BMI are strongly associated with increased "
            f"cardiovascular risk. Regular physical activity (150+ min/week), maintaining "
            f"a healthy weight, and managing stress can reduce risk by up to 50%."
        )

    # ── General / catch-all ────────────────────────────────────────
    return (
        f"Based on {stats['n']} relevant patient records retrieved from the knowledge base: "
        f"{stats['n_high']}/{stats['n']} ({stats['pct_high']:.0f}%) have diagnosed heart disease. "
        f"Key observations: {risk_phrase}. "
        f"Average age: {stats['avg_age']:.0f} years, average BMI: {stats['avg_bmi']:.1f}. "
        f"{'The high prevalence among similar patients suggests significant cardiovascular concern. ' if stats['pct_high'] > 50 else ''}"
        f"Both physical biomarkers and mental health indicators should be considered "
        f"for comprehensive health assessment."
    )


# ── Public API ─────────────────────────────────────────────────────────

def ask(query: str, top_k: int = 5) -> Dict:
    """
    Full RAG pipeline: retrieve -> generate -> return response + sources.
    """
    contexts = retrieve_context(query, top_k=top_k)
    response = generate_response(query, contexts)

    return {
        "query": query,
        "response": response,
        "sources": [
            {
                "patient_id": c["patient_id"],
                "similarity": round(1 - c["distance"], 4),
                "age": c["metadata"].get("age"),
                "risk": "High" if c["metadata"].get("target") == 1 else "Low",
            }
            for c in contexts
        ],
        "num_sources": len(contexts),
    }


if __name__ == "__main__":
    print("Testing RAG pipeline ...\n")
    test_queries = [
        "What health risks does this patient have?",
        "What health risks does a 55-year-old male with high cholesterol have?",
        "Tell me about patients with high stress and anxiety levels",
        "What causes heart disease?",
        "How can I prevent heart disease?",
        "How many patients have heart disease?",
        "What about this patient's mental health?",
    ]
    for q in test_queries:
        print(f"Q: {q}")
        result = ask(q)
        print(f"A: {result['response']}")
        print(f"   Sources: {[s['patient_id'] for s in result['sources']]}\n")
