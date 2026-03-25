"""
Run all 4 pipelines on the dev set (or a subset), compute accuracy per pipeline, print or save results.
Do not remove or rename this file.
"""

# TODO: Implement.
# 1. Build or load the global index (corpus.build_index or corpus.load_index).
# 2. Load evaluation examples via data_loader (query, answer, alt_ans per row).
# 3. For each example: run each of the 4 pipelines (RAG Fusion, HyDE, CRAG, Graph RAG; each retrieves from the global index), get predicted answer.
# 4. Evaluate via evaluation.py (compare prediction to answer/alt_ans), aggregate accuracy per pipeline.
# 5. Print or save results (e.g. accuracy per pipeline).
"""
Run all 4 pipelines on the dev set (or a subset), compute accuracy per pipeline, print or save results.
Do not remove or rename this file.
"""
import os
import json
import time
from src.corpus import build_index, load_index
from src.data_loader import load_examples
from src.evaluation import evaluate_prediction, calculate_metrics

# Import the 4 pipelines
from src.pipelines.rag_fusion import run_rag_fusion
from src.pipelines.hyde import run_hyde
from src.pipelines.crag import run_crag
from src.pipelines.graph_rag import run_graph_rag

# Configuration paths
DATASET_PATH = "dataset/crag_task_1_and_2_dev_v4.jsonl"
INDEX_SAVE_PATH = "dataset/global_index.pkl"
RESULTS_SAVE_PATH = "evaluation_results.json"

def main():
    print("=== Phase 1: Preparing Global Index ===")
    # Build or load the index once
    if os.path.exists(INDEX_SAVE_PATH):
        index = load_index(INDEX_SAVE_PATH)
    else:
        index = build_index(DATASET_PATH, save_path=INDEX_SAVE_PATH)
        
    print("\n=== Phase 2: Running Evaluation on Dev Set ===")
    
    # Dictionaries to track scores for each pipeline
    pipeline_results = {
        "rag_fusion": [],
        "hyde": [],
        "crag": [],
        "graph_rag": []
    }
    
    # For testing, you might want to add limit=10 to load_examples during development
    # e.g., load_examples(path=DATASET_PATH, limit=10)
    dev_examples = list(load_examples(path=DATASET_PATH, limit=10))
    total_examples = len(dev_examples)
    
    print(f"Loaded {total_examples} examples. Starting evaluation...\n")
    
    for i, example in enumerate(dev_examples):
        query = example["query"]
        answer = example["answer"]
        alt_ans = example["alt_ans"]
        
        print(f"--- Example {i+1}/{total_examples} ---")
        print(f"Q: {query}")
        print(f"Gold Answer: {answer}")
        
        # 1. RAG Fusion
        print("\n[RAG Fusion]")
        rf_ans, rf_ctx, rf_score = run_rag_fusion(query, index)
        rf_eval = evaluate_prediction(rf_ans, answer, alt_ans)
        pipeline_results["rag_fusion"].append(rf_eval)
        print(f"Score/Confidence: {rf_score:.4f}")
        print(f"Predicted: {rf_ans}")
        print(f"Match: {'Yes' if rf_eval else 'No'}")
        
        # 2. HyDE
        print("\n[HyDE]")
        hyde_ans, hyde_ctx, hyde_score = run_hyde(query, index)
        hyde_eval = evaluate_prediction(hyde_ans, answer, alt_ans)
        pipeline_results["hyde"].append(hyde_eval)
        print(f"Score/Confidence: {hyde_score:.4f}")
        print(f"Predicted: {hyde_ans}")
        print(f"Match: {'Yes' if hyde_eval else 'No'}")
        
        # 3. CRAG
        print("\n[CRAG]")
        crag_ans, crag_ctx, crag_score = run_crag(query, index)
        crag_eval = evaluate_prediction(crag_ans, answer, alt_ans)
        pipeline_results["crag"].append(crag_eval)
        print(f"Score/Confidence: {crag_score:.4f}")
        print(f"Predicted: {crag_ans}")
        print(f"Match: {'Yes' if crag_eval else 'No'}")
        
        # 4. Graph RAG
        # Note: We are using our document similarity graph spreading here
        print("\n[Graph RAG]")
        gr_ans, gr_ctx, gr_score = run_graph_rag(query, index)
        gr_eval = evaluate_prediction(gr_ans, answer, alt_ans)
        pipeline_results["graph_rag"].append(gr_eval)
        print(f"Score/Confidence: {gr_score:.4f}")
        print(f"Predicted: {gr_ans}")
        print(f"Match: {'Yes' if gr_eval else 'No'}")
        
        print("-" * 50)

        time.sleep(1) # Optional: add a small delay between examples for readability and to avoid hitting rate limits during development

    print("\n=== Phase 3: Final Results ===")
    final_metrics = {}
    
    for pipeline_name, results in pipeline_results.items():
        metrics = calculate_metrics(results)
        final_metrics[pipeline_name] = metrics
        print(f"{pipeline_name.upper()}:")
        print(f"  Accuracy: {metrics['accuracy'] * 100:.2f}% ({metrics['correct']}/{metrics['total']})")

    # Save the output to a file to satisfy the deliverable requirement
    with open(RESULTS_SAVE_PATH, "w") as f:
        json.dump(final_metrics, f, indent=4)
        
    print(f"\nResults successfully saved to {RESULTS_SAVE_PATH}")

if __name__ == "__main__":
    main()