import re
from typing import Any, Dict, Iterable

try:
    import pandas as pd  # optional; used if available
except Exception:
    pd = None


def summarize_results(dataset_name: str, evaluation_results, source_examples=None, *, group_by: str | None = None, where: Dict[str, Any] | None = None):
    """
    Simple, readable summary of evaluation results.
    - Supports inputs as DataFrame, dict-of-lists, or iterable.
    - Optional filtering via `where={col: value}` and grouping via `group_by`.
    """

    def normalize_label(value: Any) -> str:
        return re.sub(r"[^a-z]", "", str(value).strip().lower())

    # 1) Build a minimal DataFrame when possible
    results_df = None
    labels_sequence = []

    if hasattr(evaluation_results, "columns"):
        results_df = evaluation_results.copy()
    elif isinstance(evaluation_results, dict) and pd is not None:
        results_df = pd.DataFrame(evaluation_results)
    else:
        try:
            labels_sequence = list(evaluation_results or [])
            if pd is not None:
                results_df = pd.DataFrame({"label": labels_sequence})
        except TypeError:
            labels_sequence = []

    # 2) Ensure we have a label column and a normalized label
    if results_df is not None:
        if "label" not in results_df.columns:
            for alt in ("prediction", "classification", "result"):
                if alt in results_df.columns:
                    results_df["label"] = results_df[alt]
                    break
        if "label" in results_df.columns:
            results_df["label_norm"] = results_df["label"].astype(str).map(normalize_label)

    # 3) Optionally enrich with group_by column from source_examples
    if results_df is not None and group_by and group_by not in results_df.columns and source_examples is not None:
        if isinstance(source_examples, Iterable):
            try:
                if pd is not None:
                    src = pd.DataFrame(list(source_examples))
                    if group_by in src.columns:
                        results_df[group_by] = src[group_by].iloc[: len(results_df)].values
            except Exception:
                pass

    # 4) Apply simple filters
    if results_df is not None and where:
        for key, value in where.items():
            if key in results_df.columns:
                results_df = results_df[results_df[key] == value]

    # 5) Compute totals and counts
    if results_df is not None and "label_norm" in results_df.columns:
        total = len(results_df)
        label_counts_by_value = results_df["label_norm"].value_counts().to_dict()
        grounded = label_counts_by_value.get("grounded", 0)
        hallucination = label_counts_by_value.get("hallucination", 0)
    else:
        # No DataFrame path: tally from simple list
        total = len(labels_sequence)
        grounded = sum(1 for x in labels_sequence if "grounded" in normalize_label(x))
        hallucination = sum(1 for x in labels_sequence if "hallucination" in normalize_label(x))

    rate = (hallucination / total) if total else 0
    verdict = "Good" if rate < 0.2 else ("OK" if rate < 0.4 else "Poor")

    # 6) Print clean summary
    print(f"{dataset_name}:")
    print(f"- total: {total}")
    print(f"- grounded: {grounded}")
    print(f"- hallucination: {hallucination}")
    print(f"- rate: {rate:.2f}")
    print(f"- verdict: {verdict}")

    # 7) Optional grouping
    if results_df is not None and group_by and group_by in results_df.columns and "label_norm" in results_df.columns:
        print(f"- breakdown by {group_by}:")
        grouped_counts_dataframe = (
            results_df.groupby(group_by)["label_norm"].value_counts().unstack(fill_value=0).reset_index()
        )
        for _, group_row in grouped_counts_dataframe.iterrows():
            group_value = group_row[group_by]
            grounded_count = int(group_row.get("grounded", 0))
            hallucination_count = int(group_row.get("hallucination", 0))
            total_count = grounded_count + hallucination_count
            hallucination_rate = (hallucination_count / total_count) if total_count else 0
            print(f"  * {group_by}={group_value}: n={total_count} grounded={grounded_count} hallucination={hallucination_count} rate={hallucination_rate:.2f}")


