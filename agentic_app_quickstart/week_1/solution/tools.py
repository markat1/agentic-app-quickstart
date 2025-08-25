from typing import Any
from agents import function_tool
from agents.run_context import RunContextWrapper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time
from phoenix.trace import tracer

@tracer.chain
def default_tool_error(ctx: RunContextWrapper[pd.DataFrame], error: Exception) -> str:
    base = str(error)
    try:
        df = ctx.context
        if isinstance(df, pd.DataFrame):
            cols = ", ".join(df.columns.tolist())

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            try:
                coerced = df.apply(pd.to_numeric, errors="coerce")
                coercible_cols = coerced.columns[coerced.notna().any()].tolist()
            except Exception:
                coercible_cols = []

            numeric_like_set = set(numeric_cols) | set(coercible_cols)
            numeric_like = [c for c in df.columns.tolist() if c in numeric_like_set]
            hint = f" numeric-like: {', '.join(numeric_like)}" if numeric_like else ""
            return f"{base}. columns: [{cols}]{hint}"
    except Exception:
        pass
    return base

@tracer.chain
def data_validation_error(ctx: RunContextWrapper[pd.DataFrame], error: Exception) -> str:
    """
    Specialized error handler for data validation issues.
    Provides helpful suggestions for common data problems.
    """
    error_msg = str(error)
   
    try:
        df = ctx.context
        if isinstance(df, pd.DataFrame):
            available_cols = list(df.columns)
           
            if "not found" in error_msg.lower():
                return f"Column not found. Available columns are: {', '.join(available_cols)}"
            elif "no numeric data" in error_msg.lower():
                return f"No numeric data found. Available columns are: {', '.join(available_cols)}"
            elif "must pass" in error_msg.lower():
                return f"Missing required parameter. Please provide all required arguments."
            else:
                return f"Data error: {error_msg}. Available columns: {', '.join(available_cols)}"
    except Exception:
        pass
   
    return f"Data validation error: {error_msg}"


@function_tool(failure_error_function=default_tool_error)
def calculate_column_average(ctx: RunContextWrapper[pd.DataFrame], column_name: str) -> float:
    """
    A tool function that calculates the average of a specified column.
    """
    df: pd.DataFrame = ctx.context
    return float(df[column_name].mean())


@function_tool(failure_error_function=default_tool_error)
def calculate_filtered_column_average(
    ctx: RunContextWrapper[pd.DataFrame],
    filter_column: str,
    value: Any,
    target_column: str | None = None,
) -> float:
    """
    Calculates the mean of target_column for rows where filter_column == value.
    """
    if not target_column:
        raise ValueError("You must pass in 'target_column' for the given filter")
    df = ctx.context
    mask = df[filter_column] == value
    return float(df.loc[mask, target_column].mean())


@function_tool(failure_error_function=default_tool_error)
def count_rows_with_value(
    ctx: RunContextWrapper[pd.DataFrame], column_name: str, value: Any
) -> int:
    """
    Counts the number of rows in a DataFrame where column_name equals value.
    """
    df: pd.DataFrame = ctx.context
    return int((df[column_name] == value).sum())


@function_tool(failure_error_function=default_tool_error)
def get_column_names(ctx: RunContextWrapper[pd.DataFrame]) -> list[str]:
    """
    Retrieves the names of all columns in a DataFrame.
    """
    df = ctx.context
    return df.columns.tolist()


@function_tool(failure_error_function=data_validation_error)
def find_column_maximum(ctx: RunContextWrapper[pd.DataFrame], column_name: str) -> float:
    """
    Finds the maximum value in the specified column.
    """
    df = ctx.context
    numeric_data = pd.to_numeric(df[column_name], errors='coerce').dropna()
    return float(numeric_data.max())


@function_tool(failure_error_function=data_validation_error)
def find_column_minimum(ctx: RunContextWrapper[pd.DataFrame], column_name: str) -> float:
    """
    Finds the minimum value in the specified column.
    """
    df = ctx.context
    numeric_data = pd.to_numeric(df[column_name], errors='coerce').dropna()
    return float(numeric_data.min())


@function_tool(failure_error_function=data_validation_error)
def calculate_percentage(ctx: RunContextWrapper[pd.DataFrame], column_name: str, value: Any) -> float:
    """
    Calculates the percentage of rows where the specified column equals the given value.
    Returns the percentage as a float (0-100).
    """
    df = ctx.context
    total_rows = len(df)
    matching_rows = len(df[df[column_name] == value])
    percentage = (matching_rows / total_rows) * 100
    return float(percentage)


@function_tool(failure_error_function=data_validation_error)
def calculate_ratio(ctx: RunContextWrapper[pd.DataFrame], column_name: str, value1: Any, value2: Any) -> float:
    """
    Calculates the ratio between two values in a column.
    Returns the ratio as a float.
    """
    df = ctx.context
    count1 = len(df[df[column_name] == value1])
    count2 = len(df[df[column_name] == value2])
   
    if count2 == 0:
        raise ValueError(f"No rows found with value '{value2}' in column '{column_name}'")
   
    ratio = count1 / count2
    return float(ratio)


@function_tool(failure_error_function=data_validation_error)
def calculate_column_percentile(ctx: RunContextWrapper[pd.DataFrame], column_name: str, percentile: float) -> float:
    """
    Calculates the value at a specific percentile in a column.
    Percentile should be between 0 and 100.
    """
    df = ctx.context
    if not 0 <= percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100")
   
    numeric_data = pd.to_numeric(df[column_name], errors='coerce').dropna()
    result = numeric_data.quantile(percentile / 100)
    return float(result)


@function_tool(failure_error_function=data_validation_error)
def find_outlier_limits(ctx: RunContextWrapper[pd.DataFrame], column_name: str, fold: float = 1.5) -> dict:
    """
    Finds outlier limits using the IQR method.
    Returns a dictionary with lower_limit, upper_limit, and outlier_count.
    """
    df = ctx.context
    numeric_data = pd.to_numeric(df[column_name], errors='coerce').dropna()
   
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)
    IQR = Q3 - Q1
   
    lower_limit = Q1 - (IQR * fold)
    upper_limit = Q3 + (IQR * fold)
   
    # Count outliers
    outliers = numeric_data[(numeric_data < lower_limit) | (numeric_data > upper_limit)]
    outlier_count = len(outliers)
   
    return {
        "lower_limit": float(lower_limit),
        "upper_limit": float(upper_limit),
        "outlier_count": int(outlier_count),
        "total_count": int(len(numeric_data)),
        "outlier_percentage": float((outlier_count / len(numeric_data)) * 100)
    }


@function_tool(failure_error_function=data_validation_error)
def remove_outliers(ctx: RunContextWrapper[pd.DataFrame], column_name: str, fold: float = 1.5, keep_outliers: bool = False) -> dict:
    """
    Removes outliers from the dataset using IQR method.
    Returns statistics about the operation and the cleaned data.
    """
    df = ctx.context.copy()
    numeric_data = pd.to_numeric(df[column_name], errors='coerce').dropna()
   
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)
    IQR = Q3 - Q1
   
    lower_limit = Q1 - (IQR * fold)
    upper_limit = Q3 + (IQR * fold)
   
    # Create mask for inliers/outliers
    if keep_outliers:
        mask = (numeric_data >= lower_limit) & (numeric_data <= upper_limit)
        removed_count = len(numeric_data) - mask.sum()
        result_df = df[mask]
    else:
        mask = (numeric_data < lower_limit) | (numeric_data > upper_limit)
        removed_count = mask.sum()
        result_df = df[~mask]
   
    return {
        "original_count": int(len(df)),
        "final_count": int(len(result_df)),
        "removed_count": int(removed_count),
        "lower_limit": float(lower_limit),
        "upper_limit": float(upper_limit),
        "fold_multiplier": float(fold),
        "cleaned_data": result_df
    }


@function_tool(failure_error_function=data_validation_error)
def find_correlation(ctx: RunContextWrapper[pd.DataFrame], column1: str, column2: str) -> dict:
    """
    Finds the correlation between two numeric columns.
    Returns correlation coefficient and interpretation.
    """
    df = ctx.context
   
    # Convert both columns to numeric, dropping NaN values
    col1_data = pd.to_numeric(df[column1], errors='coerce').dropna()
    col2_data = pd.to_numeric(df[column2], errors='coerce').dropna()
   
    # Align the data by index to handle different NaN patterns
    aligned_data = pd.concat([col1_data, col2_data], axis=1).dropna()
   
    if len(aligned_data) < 2:
        raise ValueError(f"Insufficient data for correlation analysis. Need at least 2 valid pairs.")
   
    # Calculate correlation
    correlation = aligned_data[column1].corr(aligned_data[column2])
   
    # Interpret correlation strength
    if abs(correlation) >= 0.8:
        strength = "very strong"
    elif abs(correlation) >= 0.6:
        strength = "strong"
    elif abs(correlation) >= 0.4:
        strength = "moderate"
    elif abs(correlation) >= 0.2:
        strength = "weak"
    else:
        strength = "very weak"
   
    # Determine direction
    direction = "positive" if correlation > 0 else "negative" if correlation < 0 else "no"
   
    return {
        "correlation_coefficient": float(correlation),
        "strength": strength,
        "direction": direction,
        "interpretation": f"There is a {strength} {direction} correlation between {column1} and {column2} (r = {correlation:.3f})",
        "data_points_used": int(len(aligned_data))
    }


@function_tool(failure_error_function=data_validation_error)
def get_correlation_matrix(ctx: RunContextWrapper[pd.DataFrame], columns: list[str] | None = None) -> dict:
    """
    Creates a correlation matrix for numeric columns.
    If no columns specified, uses all numeric columns.
    """
    df = ctx.context
   
    # Select numeric columns
    if columns:
        # Filter to specified columns and convert to numeric
        numeric_df = df[columns].apply(pd.to_numeric, errors='coerce')
    else:
        # Use all numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
   
    # Drop rows with any NaN values for correlation calculation
    clean_df = numeric_df.dropna()
   
    if len(clean_df) < 2:
        raise ValueError("Insufficient data for correlation matrix. Need at least 2 rows of numeric data.")
   
    if len(clean_df.columns) < 2:
        raise ValueError("Need at least 2 numeric columns for correlation analysis.")
   
    # Calculate correlation matrix
    corr_matrix = clean_df.corr()
   
    # Convert to dictionary format for easier handling
    corr_dict = {}
    for col1 in corr_matrix.columns:
        corr_dict[col1] = {}
        for col2 in corr_matrix.columns:
            corr_dict[col1][col2] = float(corr_matrix.loc[col1, col2])
   
    return {
        "correlation_matrix": corr_dict,
        "columns_analyzed": list(corr_matrix.columns),
        "data_points_used": int(len(clean_df)),
        "summary": f"Correlation matrix for {len(corr_matrix.columns)} columns using {len(clean_df)} data points"
    }


@function_tool(failure_error_function=default_tool_error)
def create_histogram(ctx: RunContextWrapper[pd.DataFrame], column_name: str, title: str | None = None) -> str:
    """
    Creates a histogram of the specified column and saves it as a PNG file.
    Returns the filename of the saved plot.
    """
    df: pd.DataFrame = ctx.context
   
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame")
   
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
   
    # Generate filename with timestamp
    timestamp = int(time())
    filename = f"hist_{column_name}_{timestamp}.png"
    filepath = os.path.join("plots", filename)
   
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.hist(df[column_name].dropna(), bins=20, edgecolor='black', alpha=0.7)
    plt.title(title or f"{column_name.title()} Distribution")
    plt.xlabel(column_name.title())
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
   
    # Save and close
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
   
    return f"Histogram saved as {filename}"


@function_tool(failure_error_function=default_tool_error)
def create_bar_chart(ctx: RunContextWrapper[pd.DataFrame], x_column: str, y_column: str, title: str | None = None) -> str:
    """
    Creates a bar chart of x_column vs y_column and saves it as a PNG file.
    Returns the filename of the saved plot.
    """
    df: pd.DataFrame = ctx.context
    
    if x_column not in df.columns:
        raise ValueError(f"Column '{x_column}' not found in the DataFrame")
    if y_column not in df.columns:
        raise ValueError(f"Column '{y_column}' not found in the DataFrame")
    
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = int(time())
    filename = f"bar_{x_column}_vs_{y_column}_{timestamp}.png"
    filepath = os.path.join("plots", filename)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Group by x_column and calculate mean of y_column
    grouped_data = df.groupby(x_column)[y_column].mean().sort_values(ascending=False)
    
    plt.bar(range(len(grouped_data)), grouped_data.values, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(title or f"Average {y_column.title()} by {x_column.title()}")
    plt.xlabel(x_column.title())
    plt.ylabel(f"Average {y_column.title()}")
    plt.xticks(range(len(grouped_data)), grouped_data.index, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Save and close
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return f"Bar chart saved as {filename}"


@function_tool(failure_error_function=default_tool_error)
def create_scatter_plot(ctx: RunContextWrapper[pd.DataFrame], x_column: str, y_column: str, title: str | None = None) -> str:
    """
    Creates a scatter plot of x_column vs y_column and saves it as a PNG file.
    Returns the filename of the saved plot.
    """
    df: pd.DataFrame = ctx.context
   
    if x_column not in df.columns:
        raise ValueError(f"Column '{x_column}' not found in the DataFrame")
    if y_column not in df.columns:
        raise ValueError(f"Column '{y_column}' not found in the DataFrame")
   
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
   
    # Generate filename with timestamp
    timestamp = int(time())
    filename = f"scatter_{x_column}_vs_{y_column}_{timestamp}.png"
    filepath = os.path.join("plots", filename)
   
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_column], df[y_column], alpha=0.6, edgecolors='black', linewidth=0.5)
    plt.title(title or f"{x_column.title()} vs {y_column.title()}")
    plt.xlabel(x_column.title())
    plt.ylabel(y_column.title())
    plt.grid(True, alpha=0.3)
   
    # Save and close
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
   
    return f"Scatter plot saved as {filename}"


@function_tool(failure_error_function=data_validation_error)
def compare_with_dataset(ctx: RunContextWrapper[pd.DataFrame], target_dataset: str) -> dict:
    """
    Simple comparison of current dataset with another dataset.
    """
    df_current = ctx.context
   
    # Load target dataset
    dataset_paths = {"employees": "data/employee_data.csv", "weather": "data/weather_data.csv", "sales": "data/sample_sales.csv"}
   
    if target_dataset not in dataset_paths:
        raise ValueError(f"Dataset '{target_dataset}' not found. Available: {list(dataset_paths.keys())}")
   
    df_target = pd.read_csv(dataset_paths[target_dataset])
   
    return {
        "current": {"rows": len(df_current), "columns": list(df_current.columns)},
        "target": {"rows": len(df_target), "columns": list(df_target.columns)},
        "common_columns": list(set(df_current.columns) & set(df_target.columns))
    }

@function_tool(failure_error_function=default_tool_error)
def load_data(ctx: RunContextWrapper[pd.DataFrame], file_path: str) -> str:
    """
    Loads data from CSV, JSON, or Excel file into a pandas DataFrame and updates the context.
    Args:
        ctx (RunContextWrapper[Any]): The context wrapper to update.
        file_path (str): Path to the data file.
    Returns:
        str: Success message with shape info.
    Raises:
        ValueError: If file format is unsupported or file cannot be loaded.
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please use CSV, JSON, or Excel files.")
    except Exception as e:
        raise ValueError(f"Failed to load file '{file_path}': {e}")
    ctx.context = df
    return f"Loaded {file_path} with shape {df.shape}"

@function_tool(failure_error_function=default_tool_error)
def compare_csv_files(ctx: RunContextWrapper[pd.DataFrame], file_paths: list[str]) -> dict:
    """
    Compares multiple CSV files: returns row/column counts and common columns.
    Stores the result in ctx.context.
    Args:
        ctx (RunContextWrapper[Any]): Context wrapper to update.
        file_paths (list[str]): List of CSV file paths.
    Returns:
        dict: Summary of each file and common columns.
    """
    summaries = {}
    columns_sets = []

    for path in file_paths:
        try:
            df = pd.read_csv(path)
            summaries[path] = {
                "rows": len(df),
                "columns": list(df.columns)
            }
            columns_sets.append(set(df.columns))
        except Exception as e:
            summaries[path] = {"error": str(e)}

    # Find common columns across all files
    common_columns = list(set.intersection(*columns_sets)) if columns_sets else []

    result = {
        "file_summaries": summaries,
        "common_columns": common_columns,
        "file_count": len(file_paths)
    }
    ctx.context = result
    return result