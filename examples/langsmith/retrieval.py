"""Minimal lexical retrieval for the Acme Cloud support benchmark."""

import re
from typing import Dict, List, Set

from knowledge_base import SUPPORT_DOCS


def _tokenize(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def retrieve_documents(subject: str, message: str, top_k: int = 2) -> List[Dict]:
    query = f"{subject}\n{message}"
    query_tokens = _tokenize(query)
    query_text = query.lower()
    scored_docs = []

    for doc in SUPPORT_DOCS:
        title_tokens = _tokenize(doc["title"])
        body_tokens = _tokenize(doc["body"])
        keyword_tokens = set()
        for keyword in doc.get("keywords", []):
            keyword_tokens |= _tokenize(keyword)

        overlap_score = len(query_tokens & body_tokens)
        title_score = 2 * len(query_tokens & title_tokens)
        keyword_score = 3 * len(query_tokens & keyword_tokens)
        phrase_bonus = sum(3 for keyword in doc.get("keywords", []) if keyword.lower() in query_text)

        total_score = overlap_score + title_score + keyword_score + phrase_bonus
        scored_docs.append((total_score, doc))

    scored_docs.sort(key=lambda item: (item[0], item[1]["doc_id"]), reverse=True)
    return [doc for score, doc in scored_docs[:top_k] if score > 0] or [doc for _, doc in scored_docs[:top_k]]

