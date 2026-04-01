def select_diseases(disease_probs, threshold=0.1, min_k=5):
    """
    Select clinically relevant diseases based on probability.

    - Uses threshold when possible
    - Falls back to Top-K to guarantee RAG context
    - Preserves original return type and sorting
    """

    # First: threshold-based selection (original logic)
    selected = {
        disease: prob
        for disease, prob in disease_probs.items()
        if prob >= threshold
    }

    # Fallback: ensure minimum number of diseases for RAG
    if len(selected) < min_k:
        sorted_all = sorted(
            disease_probs.items(),
            key=lambda x: x[1],
            reverse=True
        )
        selected = dict(sorted_all[:min_k])

    # Final sort (descending probability)
    return dict(
        sorted(selected.items(), key=lambda x: x[1], reverse=True)
    )
