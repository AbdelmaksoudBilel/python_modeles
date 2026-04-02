from src.rag.retrieval_service import HybridRetriever

retriever = HybridRetriever()

question = "Comment calmer un enfant autiste pendant une crise ?"

results = retriever.search(question, k=5)

for i, r in enumerate(results):

    print("\nResult", i+1)
    print("Source :", r["source"])
    print("Text :", r["text"][:200])