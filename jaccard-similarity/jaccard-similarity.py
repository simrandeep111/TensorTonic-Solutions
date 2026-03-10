def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    A=set_a
    B=set_b
    if not A and not B:
        return 0.0
    intersection = len(set(A) & set(B))  
    union = len(set(A) | set(B))         
    return intersection / union