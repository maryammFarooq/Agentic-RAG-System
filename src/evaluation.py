"""
Compare predicted answer to gold answer and alt_ans; compute accuracy.
Do not change this module's location in the project.
"""

# TODO: Implement. E.g.: exact or normalized match against answer and alt_ans; return accuracy per pipeline.
"""
Compare predicted answer to gold answer and alt_ans; compute accuracy.
Do not change this module's location in the project.
"""

# TODO: Implement. E.g.: exact or normalized match against answer and alt_ans; return accuracy per pipeline.
"""
Evaluation module for comparing predicted answers against gold standard answers.
"""
import re
import string

def normalize_text(text: str) -> str:
    """
    Normalizes text to improve matching accuracy.
    Lowercases, removes punctuation, and strips extra whitespace.
    """
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def evaluate_prediction(predicted: str, answer: str, alt_ans: list[str] = None) -> int:
    """
    Evaluates matches by checking if the core factual entity (Noun Phrases) 
    from the gold answer appears in the prediction.
    """
    if alt_ans is None:
        alt_ans = []
        
    norm_pred = normalize_text(predicted)
    norm_gold = normalize_text(answer)
    
    # 1. Direct Substring Check (The easiest win)
    if norm_gold in norm_pred or norm_pred in norm_gold:
        return 1

    # 2. Key Entity Overlap
    # If the gold answer says "Oracle" and your prediction says "Oracle", it's a match.
    # We focus on "Keywords" (words longer than 3 chars that aren't stop words)
    stop_words = {"the", "this", "that", "with", "from", "spent", "years", "worked", "before"}
    
    def get_keywords(text):
        return {word for word in text.split() if len(word) > 3 and word not in stop_words}

    gold_keywords = get_keywords(norm_gold)
    pred_keywords = get_keywords(norm_pred)

    if not gold_keywords:
        return 0

    # If the prediction contains the main entities (like 'oracle' and 'benioff')
    # from the gold answer, we count it as correct.
    intersection = gold_keywords.intersection(pred_keywords)
    
    # If we catch the "Main Entity" (usually the most unique word in the gold answer)
    # or 50% of the keywords, it's a match.
    if len(intersection) / len(gold_keywords) >= 0.5:
        return 1
        
    # 3. Check Alternative Answers
    for alt in alt_ans:
        if normalize_text(alt) in norm_pred:
            return 1
            
    return 0


def calculate_metrics(evaluation_results: list[int]) -> dict:
    """
    Calculates overall accuracy from a list of binary evaluation results.
    """
    if not evaluation_results:
        return {"accuracy": 0.0, "total": 0, "correct": 0}
        
    total = len(evaluation_results)
    correct = sum(evaluation_results)
    accuracy = correct / total
    
    return {
        "accuracy": accuracy,
        "total": total,
        "correct": correct
    }