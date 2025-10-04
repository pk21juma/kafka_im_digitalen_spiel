import re
import math
import numpy as np

def calculate_score(text, keywords):
    """
    Calculates a score base on Keywords and weighting.
    Uses logarithmic flattening per keyword (clipping to 100).
    """
    total_score = 0

    for kw, weight in keywords.items():
        pattern = rf"\b{re.escape(kw.lower())}\b"
        matches = re.findall(pattern, text.lower())
        num_matches = len(matches)

        if num_matches > 0:
            # Basiswert (Gewicht 1–5 zu 4–20 Punkte)
            base = weight * 4
            score_kw = base * math.log(1 + num_matches)
            total_score += score_kw

    return round(min(total_score, 100), 2)


def rescale_scores(scores, method="zscore"):
    """
    Rescaling of a number list to values 0-100.
    
    Methods:
    - "zscore": Standardization with Z-Score + Sigmoid
    - "minmax": Min-max scaling to 0-100
    """
    scores = np.array(scores, dtype=float)
    
    if method == "zscore":
        mean = np.mean(scores)
        std = np.std(scores)
        if std == 0:
            return np.full_like(scores, 50.0)
        z = (scores - mean) / std
        sig = 1 / (1 + np.exp(-z))
        return sig * 100
    
    elif method == "minmax":
        min_val, max_val = np.min(scores), np.max(scores)
        if min_val == max_val:
            return np.full_like(scores, 50.0)
        return (scores - min_val) / (max_val - min_val) * 100
    
    else:
        raise ValueError("Unknown method. Use 'zscore' or 'minmax'.")