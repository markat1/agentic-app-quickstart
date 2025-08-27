import re
from typing import Any, Dict, Iterable, Tuple
import pandas as pd


def _normalize_label(value: Any) -> str:
    """Return a normalized label string: lowercase letters only (strip spaces/punct)."""
    return re.sub(r"[^a-z]", "", str(value).strip().lower())


def _build_results_dataframe(evaluation_results) -> pd.DataFrame:
    """Build a DataFrame with at least columns: label, label_norm.

    Accepts: pandas DataFrame, dict-of-lists, or any iterable of labels. If a
    label column is missing, tries common fallbacks: prediction/classification/result.
    """
    if isinstance(evaluation_results, pd.DataFrame):
        df = evaluation_results.copy()
    elif isinstance(evaluation_results, dict):
        df = pd.DataFrame(evaluation_results)
    else:
        labels = list(evaluation_results or [])
        df = pd.DataFrame({"label": labels})
    if "label" not in df.columns:
        for alt in ("prediction", "classification", "result"):
            if alt in df.columns:
                df["label"] = df[alt]
                break
    df["label_norm"] = df["label"].astype(str).map(_normalize_label)
    return df[df["label_norm"].ne("")]


def _attach_group_column(results_df: pd.DataFrame, source_examples, group_by: str | None) -> pd.DataFrame:
    """Attach a grouping column from source_examples if not present on results_df."""
    if not group_by or group_by in results_df.columns or not isinstance(source_examples, Iterable):
        return results_df
    src_df = pd.DataFrame(list(source_examples))
    if group_by in src_df.columns:
        results_df[group_by] = src_df[group_by].iloc[: len(results_df)].values
    return results_df


def _apply_filters(results_df: pd.DataFrame, where: Dict[str, Any] | None) -> pd.DataFrame:
    """Apply simple equality filters (where) to the results DataFrame, if provided."""
    if not where:
        return results_df
    for key, value in where.items():
        if key in results_df.columns:
            results_df = results_df[results_df[key] == value]
    return results_df


def _compute_overall_metrics(results_df: pd.DataFrame) -> Tuple[int, int, int, float, str]:
    """Compute total, grounded, hallucination, rate, and verdict from results_df."""
    total = len(results_df)
    counts = results_df["label_norm"].value_counts().to_dict()
    grounded = counts.get("grounded", 0)
    hallucination = counts.get("hallucination", 0)
    rate = (hallucination / total) if total else 0.0
    verdict = "Good" if rate < 0.2 else ("OK" if rate < 0.4 else "Poor")
    return total, grounded, hallucination, rate, verdict


def _print_overall(dataset_name: str, total: int, grounded: int, hallucination: int, rate: float, verdict: str) -> None:
    """Pretty-print the overall summary numbers and verdict for a dataset."""
    print(f"{dataset_name}:")
    print(f"- total: {total}")
    print(f"- grounded: {grounded}")
    print(f"- hallucination: {hallucination}")
    print(f"- rate: {rate:.2f}")
    print(f"- verdict: {verdict}")


def _print_group_breakdown(results_df: pd.DataFrame, group_by: str | None) -> None:
    """Pretty-print per-group grounded/hallucination counts and rates if grouping is enabled."""
    if not group_by or group_by not in results_df.columns:
        return
    print(f"- breakdown by {group_by}:")
    grouped = results_df.groupby(group_by)["label_norm"].value_counts().unstack(fill_value=0).reset_index()
    for _, row in grouped.iterrows():
        group_value = row[group_by]
        grounded_count = int(row.get("grounded", 0))
        hallucination_count = int(row.get("hallucination", 0))
        total_count = grounded_count + hallucination_count
        hallucination_rate = (hallucination_count / total_count) if total_count else 0
        print(
            f"  * {group_by}={group_value}: n={total_count} grounded={grounded_count} "
            f"hallucination={hallucination_count} rate={hallucination_rate:.2f}"
        )


def summarize_results(dataset_name: str, evaluation_results, source_examples=None, *, group_by: str | None = None, where: Dict[str, Any] | None = None):
    """Print a concise summary of evaluation labels with optional filter and group-by.

    - dataset_name: label for the dataset being summarized (header only)
    - evaluation_results: DataFrame/dict/iterable produced by the evaluator
    - source_examples: optional list of dicts to supply grouping columns (e.g., agent_name)
    - group_by: optional column name to break down results by
    - where: optional mapping of column->value to filter results before counting
    """
    results_df = _build_results_dataframe(evaluation_results)
    results_df = _attach_group_column(results_df, source_examples, group_by)
    results_df = _apply_filters(results_df, where)
    total, grounded, hallucination, rate, verdict = _compute_overall_metrics(results_df)
    _print_overall(dataset_name, total, grounded, hallucination, rate, verdict)
    _print_group_breakdown(results_df, group_by)


