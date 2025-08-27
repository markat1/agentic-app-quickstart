import re
def summarize_results(dataset_name: str, evaluation_results, source_examples=None):
    """Shows item count and label distribution. Labels: grounded (supported by input) and hallucination (unsupported additions/changes)."""
    # Count labels
    label_counts = {}
    
    # Normalize results into a list of labels or rows
    # Case 1: pandas DataFrame
    if hasattr(evaluation_results, 'empty') and evaluation_results.empty:
        data_to_process = []
        labels_series = []
    elif hasattr(evaluation_results, 'columns') and ('label' in getattr(evaluation_results, 'columns', [])):
        labels_series = list(evaluation_results['label'])
        data_to_process = labels_series
    # Case 2: dict-of-lists (phoenix.evals typical shape)
    elif isinstance(evaluation_results, dict):
        labels_series = evaluation_results.get('label') or evaluation_results.get('labels') or []
        data_to_process = list(labels_series)
    # Case 3: None or other iterables
    elif evaluation_results is None:
        data_to_process = []
    else:
        # Materialize iterator/generator so we can count and iterate reliably
        try:
            data_to_process = list(evaluation_results)
        except TypeError:
            data_to_process = []
        labels_series = []
    
    for row in data_to_process:
        label = None
        if isinstance(row, str):
            label = row
        
        if isinstance(row, dict):
            label = (
                row.get("label")
                or row.get("prediction")
                or row.get("classification")
                or row.get("result")
            )
        else:
            # Handle objects returned by eval libraries (e.g., phoenix.evals)
            for attr in ("label", "prediction", "classification", "result"):
                if hasattr(row, attr):
                    label = getattr(row, attr)
                    break
            # Some objects store values under .value or .choice
            if label is None and hasattr(row, "value"):
                label = getattr(row, "value")
        if label:
            label_str = str(label).strip().lower()
            # remove punctuation/spaces to handle variants like "Grounded." or "Hallucination\n"
            label_norm = re.sub(r"[^a-z]", "", label_str)
            label_counts[label_norm] = label_counts.get(label_norm, 0) + 1
    
    # Print summary
    # Compute total robustly
    total = 0
    # Prefer label list length if available
    if isinstance(evaluation_results, dict) and (evaluation_results.get('label') or evaluation_results.get('labels')):
        seq = evaluation_results.get('label') or evaluation_results.get('labels') or []
        try:
            total = len(seq)
        except Exception:
            total = 0
    elif hasattr(evaluation_results, 'columns') and ('label' in getattr(evaluation_results, 'columns', [])):
        try:
            total = len(evaluation_results['label'])
        except Exception:
            total = 0
    else:
        total = len(data_to_process)

    if source_examples is not None and not total:
        try:
            total = len(source_examples)
        except Exception:
            pass
    # If labels are embedded in free text, count by substring
    grounded = label_counts.get("grounded", 0)
    hallucination = label_counts.get("hallucination", 0)
    if grounded == 0 and hallucination == 0 and label_counts:
        grounded = sum(cnt for key, cnt in label_counts.items() if "grounded" in key)
        hallucination = sum(cnt for key, cnt in label_counts.items() if "hallucination" in key)
    rate = hallucination / total if total else 0
    verdict = "Good" if rate < 0.2 else ("OK" if rate < 0.4 else "Poor")
    print(f"{dataset_name}: n={total} grounded={grounded} hallucination={hallucination} rate={rate:.2f} verdict={verdict}")


