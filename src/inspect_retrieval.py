"""Eyeball retrieval quality: check the store is filled and look at what comes back.

This shows plain vector search with its distances, whatever SEARCH_TYPE is set to;
the ablation script compares vector and hybrid search.
"""

from config import TOP_K
from vectorstore import get_vector_db

QUESTIONS = [
    "What is the rule for grenades?",
    "How can I use a stratagem?",
    "What are the different ways to score points?",
]


def main():
    db = get_vector_db()

    # 1. Did everything get stored?
    print("Chunks in database:", db._collection.count())

    # 2. Does retrieval return sensible results?
    #    The score is a distance: lower means more similar.
    for question in QUESTIONS:
        print(f"\n=== {question}")
        for doc, score in db.similarity_search_with_score(question, k=TOP_K):
            print(f"\n- [{score:.3f}]\n{doc.page_content}")


if __name__ == "__main__":
    main()