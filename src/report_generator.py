"""
Turns (predicted condition, confidence, Grad-CAM region) into a readable
report. Tries retrieval from real OpenI report text first (if the index
exists); falls back to a clean template if the index isn't built yet or
nothing relevant is found. Always keep the template fallback -- it's what
guarantees the demo works even if the retrieval index has issues.
"""

import os
import pickle

from sklearn.metrics.pairwise import cosine_similarity


TEMPLATE = (
    "Findings suggest {condition_phrase}, most notably in the {region}. "
    "Model confidence: {confidence_pct}%. {urgency_note}"
)

CONDITION_PHRASES = {
    "No Finding": "no significant abnormality",
    "Pneumonia": "signs consistent with pneumonia",
    "Effusion": "findings consistent with pleural effusion",
    "Cardiomegaly": "cardiac silhouette enlargement (cardiomegaly)",
    "Infiltration": "an area of pulmonary infiltration",
}


def load_index(index_path="models/report_index.pkl"):
    if not os.path.exists(index_path):
        return None
    with open(index_path, "rb") as f:
        return pickle.load(f)


def retrieve_similar_report(index, condition, top_k=1):
    """Find the most similar real report mentioning this condition, if any."""
    if index is None:
        return None
    reports = index["reports"]
    mask = reports["labels"].str.contains(condition, case=False, na=False)
    if not mask.any():
        return None

    query_vec = index["vectorizer"].transform([condition])
    subset_matrix = index["tfidf_matrix"][mask.values]
    sims = cosine_similarity(query_vec, subset_matrix)[0]
    best_idx = sims.argmax()
    matched_reports = reports[mask].reset_index(drop=True)
    return matched_reports.iloc[best_idx]["text"]


def generate_report(condition, confidence, region, urgency_note="", index=None):
    condition_phrase = CONDITION_PHRASES.get(condition, condition.lower())
    confidence_pct = round(confidence * 100, 1)

    retrieved = retrieve_similar_report(index, condition) if index else None

    if retrieved:
        # Ground the retrieved real report text with our own structured facts
        # up front, so the output stays tied to this specific prediction.
        report = (
            f"AI-assisted finding: {condition_phrase}, focused in the {region} "
            f"(confidence {confidence_pct}%).\n\n"
            f"Similar documented finding pattern: {retrieved.strip()[:400]}\n\n"
            f"{urgency_note}"
        )
    else:
        report = TEMPLATE.format(
            condition_phrase=condition_phrase,
            region=region,
            confidence_pct=confidence_pct,
            urgency_note=urgency_note,
        )

    return report


if __name__ == "__main__":
    idx = load_index()
    print(generate_report("Pneumonia", 0.81, "lower right lung field", index=idx))