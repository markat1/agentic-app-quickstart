def summarize_results(dataset_name: str, evaluation_results, source_examples=None):
    """Shows item count and label distribution. Labels: grounded (supported by input) and hallucination (unsupported additions/changes)."""
    # Count labels
    label_counts = {}
    
    # Handle pandas DataFrame properly
    if hasattr(evaluation_results, 'empty') and evaluation_results.empty:
        data_to_process = []
    elif evaluation_results is None:
        data_to_process = []
    else:
        data_to_process = evaluation_results
    
    for row in data_to_process:
        if isinstance(row, dict):
            label = row.get("label") or row.get("prediction") or row.get("classification") or row.get("result")
            if label:
                label_counts[str(label)] = label_counts.get(str(label), 0) + 1
    
    # Print summary
    total = len(evaluation_results) if hasattr(evaluation_results, "__len__") else sum(1 for _ in evaluation_results or [])
    grounded = label_counts.get("grounded", 0)
    hallucination = label_counts.get("hallucination", 0)
    rate = hallucination / total if total else 0
    verdict = "Good" if rate < 0.2 else ("OK" if rate < 0.4 else "Poor")
    print(f"{dataset_name}: n={total} grounded={grounded} hallucination={hallucination} rate={rate:.2f} verdict={verdict}")


