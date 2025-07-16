import pandas as pd
import numpy as np # Import numpy for NaN values

def process_air_pollution_data(
    file_path: str,
    adm_level: int,
    metric_column: str = 'NO2_mean',
    baseline_types: list = None,
    date_col: str = 'start_date',
    output_date_col: str = 'date',
    country_name: str = None,
    lang_suffix: str = '_EN' # <--- NEW PARAMETER HERE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads air pollution data, calculates monthly and annual percentage changes
    relative to specified baselines, and returns processed DataFrames.

    Args:
        file_path (str): Path to the CSV data file.
        adm_level (int): The administrative level for grouping (0, 1, or 2).
                         Determines the 'ADM<level><lang_suffix>' column to use.
        metric_column (str, optional): The name of the pollution metric column.
                                       Defaults to 'NO2_mean'.
        baseline_types (list, optional): A list of baseline types (e.g., [2019, 'PY']).
                                         Defaults to [2019, 'PY'].
        date_col (str, optional): The original name of the date column in the CSV.
                                  Defaults to 'start_date'.
        output_date_col (str, optional): The standardized name for the date column
                                         after renaming. Defaults to 'date'.
        country_name (str, optional): The country name to assign if adm_level is 0
                                      and the ADM0 column is not naturally in the data.
                                      Defaults to None.
        lang_suffix (str, optional): The suffix for the administrative column names
                                     (e.g., '_EN' for English, '_ES' for Spanish).
                                     Defaults to '_EN'. # <--- DOCUMENTATION

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - The processed monthly DataFrame.
            - The processed annual DataFrame.

    Raises:
        ValueError: If `adm_level` is not 0, 1, or 2, or if required columns are missing.
    """
    print(f"--- Processing data for ADM level {adm_level} (suffix: {lang_suffix}) ---")

    if baseline_types is None:
        baseline_types = [2019, 'PY']

    # --- Data Loading and Initial Preparation ---
    try:
        df_monthly = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File not found at {file_path}")

    # Rename and convert date column
    if date_col != output_date_col:
        df_monthly.rename(columns={date_col: output_date_col}, inplace=True)
    df_monthly[output_date_col] = pd.to_datetime(df_monthly[output_date_col])

    # Drop 'Unnamed: 0' if it exists
    if 'Unnamed: 0' in df_monthly.columns:
        df_monthly.drop(columns=['Unnamed: 0'], inplace=True)

    # Determine the grouping column based on adm_level and the new suffix
    group_column = f'ADM{adm_level}{lang_suffix}' # <--- USE NEW PARAMETER HERE
    if group_column not in df_monthly.columns:
        # Special handling for ADM0 if it's not a pre-existing column but needs to be assigned
        if adm_level == 0 and country_name:
            df_monthly[group_column] = country_name
        else:
            raise ValueError(f"Grouping column '{group_column}' not found in DataFrame. "
                             f"Please check 'adm_level', 'lang_suffix', and data columns.")

    # --- Process Monthly Data ---
    print(f"  Calculating monthly changes for {metric_column}...")
    df_monthly_processed = df_monthly.copy()
    for b_type in baseline_types:
        df_monthly_processed = get_monthly_baseline_pc_change(
            df=df_monthly_processed,
            group_column=group_column,
            metric_column=metric_column,
            date_column=output_date_col,
            baseline_type=b_type
        )
    print("  Monthly processing complete.")

    # --- Prepare Annual Data ---
    print(f"  Aggregating to annual data for {metric_column}...")
    # Aggregate to annual means. Ensure the grouping includes the relevant ADM column.
    df_annual = df_monthly_processed.groupby([group_column, pd.Grouper(key=output_date_col, freq='YS')])[metric_column].mean().reset_index()

    # --- Process Annual Data ---
    print(f"  Calculating annual changes for {metric_column}...")
    df_annual_processed = df_annual.copy()
    for b_type in baseline_types:
        df_annual_processed = get_annual_baseline_pc_change(
            df=df_annual_processed,
            group_column=group_column,
            metric_column=metric_column,
            date_column=output_date_col,
            baseline_year_type=b_type
        )
    print("  Annual processing complete.")
    print(f"--- Processing for ADM level {adm_level} (suffix: {lang_suffix}) finished ---")

    return df_monthly_processed, df_annual_processed

def get_annual_baseline_pc_change(
    df: pd.DataFrame,
    group_column: str = 'ADM1_EN',
    baseline_year_type: str | int = 'PY', # Renamed for clarity: PY or specific year (int)
    metric_column: str = 'NO2_mean',
    date_column: str = 'date'
) -> pd.DataFrame:
    """
    Calculates the annual percentage change of a specified metric (e.g., NO2_mean)
    relative to a baseline. The baseline can be a fixed year or the previous year.

    Args:
        df (pd.DataFrame): The input DataFrame containing time-series data.
                           Expected columns: date_column (e.g., 'date'),
                           group_column (e.g., 'ADM1_EN'), NO2_column (e.g., 'NO2_mean').
        group_column (str, optional): The column name to group by (e.g., administrative unit).
                                      Defaults to 'ADM1_EN'.
        baseline_year_type (str | int, optional): Defines the baseline for comparison.
                                                - If 'PY': uses the previous year's mean as baseline.
                                                - If an int (e.g., 2019): uses the mean of that specific year as baseline.
                                                Defaults to 'PY'.
        metric_column (str, optional): The column name containing the metric for which
                                    percentage change is calculated. Defaults to 'NO2_mean'.
        date_column (str, optional): The column name containing date information.
                                     Defaults to 'date'.

    Returns:
        pd.DataFrame: The DataFrame with new columns for the calculated baseline
                      and the percentage change. Returns an empty DataFrame if
                      required columns are missing or an invalid baseline_year_type is given.

    Raises:
        ValueError: If required columns are missing or baseline_year_type is invalid.
    """
    df = df.copy() # Work on a copy to avoid modifying the original DataFrame

    # --- Input Validation ---
    required_columns = [date_column, group_column, metric_column]
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Missing required columns in DataFrame: {', '.join(missing_cols)}")

    # Ensure date_column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        try:
            df[date_column] = pd.to_datetime(df[date_column])
        except Exception as e:
            raise ValueError(f"Could not convert '{date_column}' to datetime. Error: {e}")

    df['year'] = df[date_column].dt.year

    baseline_column_name = f'baseline_{metric_column}_{baseline_year_type}'
    percent_change_column_name = f'percent_change_{metric_column}_{baseline_year_type}'

    if isinstance(baseline_year_type, int):  # Fixed baseline year case
        if baseline_year_type not in df['year'].unique():
            print(f"Warning: Baseline year {baseline_year_type} not found in data. All {percent_change_column_name} will be NaN.")
            df[baseline_column_name] = np.nan # Assign NaN if baseline year not present
        else:
            # Compute annual baseline NO2 for the specified baseline year
            baseline_df = (
                df[df['year'] == baseline_year_type]
                .groupby(group_column)[metric_column]
                .mean()
                .reset_index()
                .rename(columns={metric_column: baseline_column_name})
            )
            # Merge the fixed-year baseline into the original DataFrame
            df = df.merge(baseline_df, on=[group_column], how='left')

    elif baseline_year_type == 'PY':  # Previous year case (more efficient implementation)
        # Calculate the mean NO2_column for each group and year
        annual_group_means = df.groupby([group_column, 'year'])[metric_column].mean().reset_index()

        # Get the previous year's mean within each group using shift()
        annual_group_means[baseline_column_name] = annual_group_means.groupby(group_column)[metric_column].shift(1)

        # Merge these previous year means back into the original DataFrame
        df = df.merge(
            annual_group_means[[group_column, 'year', baseline_column_name]],
            on=[group_column, 'year'],
            how='left'
        )

    else:
        raise ValueError("Invalid 'baseline_year_type' argument. Use an integer for a fixed year or 'PY' for previous year.")

    # --- Compute the annual percentage change ---
    # Handle cases where baseline_column_name might be zero to avoid division by zero
    # Replace 0s in baseline with NaN to propagate NaNs for calculations (or choose another strategy)
    df[baseline_column_name] = df[baseline_column_name].replace(0, np.nan)

    df[percent_change_column_name] = ((df[metric_column] - df[baseline_column_name]) / df[baseline_column_name]) * 100

    return df

import pandas as pd
import numpy as np # Import numpy for NaN values

def get_monthly_baseline_pc_change(
    df: pd.DataFrame,
    group_column: str,
    metric_column: str, # Make the metric column name dynamic
    date_column: str = 'date',
    baseline_type: str | int = 'PY' # Renamed for clarity: 'PY' or specific year (int)
) -> pd.DataFrame:
    """
    Calculates the monthly percentage change of a specified metric (e.g., NO2_mean)
    relative to a baseline. The baseline can be a fixed year's monthly average
    or the previous year's monthly average for the same month.

    Args:
        df (pd.DataFrame): The input DataFrame containing time-series data.
                           Expected columns: date_column (e.g., 'date'),
                           group_column (e.g., 'ADM1_EN'), and metric_column.
        group_column (str): The column name to group by (e.g., administrative unit).
        metric_column (str): The column name containing the metric for which
                             percentage change is calculated (e.g., 'NO2_mean').
        date_column (str, optional): The column name containing date information.
                                     Defaults to 'date'.
        baseline_type (str | int, optional): Defines the baseline for comparison.
                                            - If 'PY': uses the previous year's mean for the same month as baseline.
                                            - If an int (e.g., 2019): uses the mean of that specific year for the
                                              corresponding month as baseline.
                                            Defaults to 'PY'.

    Returns:
        pd.DataFrame: The DataFrame with new columns for the calculated monthly baseline
                      and the monthly percentage change.

    Raises:
        ValueError: If required columns are missing, date_column cannot be converted
                    to datetime, or baseline_type is invalid.
    """
    df_copy = df.copy() # Work on a copy to avoid modifying the original DataFrame

    # --- Input Validation ---
    required_columns = [date_column, group_column, metric_column]
    if not all(col in df_copy.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df_copy.columns]
        raise ValueError(f"Missing required columns in DataFrame: {', '.join(missing_cols)}")

    # Ensure date_column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_column]):
        try:
            df_copy[date_column] = pd.to_datetime(df_copy[date_column])
        except Exception as e:
            raise ValueError(f"Could not convert '{date_column}' to datetime. Error: {e}")

    df_copy['year'] = df_copy[date_column].dt.year
    df_copy['month'] = df_copy[date_column].dt.month # Extract month

    baseline_column_name = f'baseline_{metric_column}_{baseline_type}'
    percent_change_column_name = f'percent_change_{metric_column}_{baseline_type}'

    if isinstance(baseline_type, int):  # Fixed baseline year case
        if baseline_type not in df_copy['year'].unique():
            print(f"Warning: Baseline year {baseline_type} not found in data. All {percent_change_column_name} will be NaN.")
            df_copy[baseline_column_name] = np.nan # Assign NaN if baseline year not present
        else:
            # Compute monthly baseline mean for the specified baseline year
            baseline_df = (
                df_copy[df_copy['year'] == baseline_type]
                .groupby([group_column, 'month'])[metric_column]
                .mean()
                .reset_index()
                .rename(columns={metric_column: baseline_column_name})
            )
            # Merge the fixed-year monthly baseline into the original DataFrame
            df_copy = df_copy.merge(baseline_df, on=[group_column, 'month'], how='left')

    elif baseline_type == 'PY':  # Previous year case (efficient implementation)
        # Calculate the mean of the metric_column for each group, year, and month
        monthly_group_means = df_copy.groupby([group_column, 'year', 'month'])[metric_column].mean().reset_index()

        # Sort to ensure correct shifting
        monthly_group_means = monthly_group_means.sort_values(by=[group_column, 'month', 'year'])

        # Get the previous year's mean for the same month within each group
        monthly_group_means[baseline_column_name] = monthly_group_means.groupby([group_column, 'month'])[metric_column].shift(1)

        # Merge these previous year monthly means back into the original DataFrame
        df_copy = df_copy.merge(
            monthly_group_means[[group_column, 'year', 'month', baseline_column_name]],
            on=[group_column, 'year', 'month'],
            how='left'
        )

    else:
        raise ValueError("Invalid 'baseline_type' argument. Use an integer for a fixed year or 'PY' for previous year.")

    # --- Compute the monthly percentage change ---
    # Handle cases where baseline_column_name might be zero to avoid division by zero
    df_copy[baseline_column_name] = df_copy[baseline_column_name].replace(0, np.nan)

    df_copy[percent_change_column_name] = ((df_copy[metric_column] - df_copy[baseline_column_name]) / df_copy[baseline_column_name]) * 100

    return df_copy