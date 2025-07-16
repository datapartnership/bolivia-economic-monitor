import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import geopandas as gpd # This is crucial for handling geometry_df as a GeoDataFrame

import pandas as pd
from bokeh.io import output_notebook, show
from bokeh.models import TabPanel, Tabs # Panel is deprecated in newer Bokeh
from bokeh.core.validation.warnings import EMPTY_LAYOUT, MISSING_RENDERERS
import bokeh.core.validation # Import the module to access silence

import pandas as pd
from bokeh.plotting import figure
from bokeh.models import Span, Legend, Label # Added Legend and Label
from bokeh.layouts import column # Added column for layout
from bokeh.palettes import Category10 # A good default color palette from Bokeh

def get_line_plot(
    dataframe: pd.DataFrame,
    title: str,
    source: str,
    subtitle: str = None,
    measure: str = "conflictIndex",
    category: str = "DT",
    event_date: str = "event_date",
    events_dict: dict = None,
    color_palette: list = None, # Added color_palette as an argument for modularity
    plot_width: int = 800,      # Added plot_width for modularity
    plot_height: int = 500      # Added plot_height for modularity
):
    """
    Create a line plot for comparing trends across different regions or categories.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing time-series data.
    title : str
        Main title for the chart.
    source : str
        Source information to display at the bottom.
    subtitle : str, optional
        Subtitle for the chart, displayed as the plot title.
    measure : str, optional
        Column name for the measure to plot on y-axis, defaults to "conflictIndex".
    category : str, optional
        Column name for grouping the data into different lines, defaults to "DT".
    event_date : str, optional
        Column name for date values, defaults to "event_date".
    events_dict : dict, optional
        Dictionary of {datetime: label} for marking significant events.
    color_palette : list, optional
        A list of colors to use for the lines. Defaults to Bokeh's Category10.
    plot_width : int, optional
        Width of the main plot in pixels. Defaults to 800.
    plot_height : int, optional
        Height of the main plot in pixels. Defaults to 500.

    Returns
    -------
    bokeh.layouts.column
        A layout containing the title, line plot, and source information.
    """
    # Use provided color_palette or a default Bokeh palette
    if color_palette is None:
        color_palette = Category10[10] # Use Category10 palette with 10 colors

    # Initialize the main plot figure
    p2 = figure(x_axis_type="datetime", width=plot_width, height=plot_height, toolbar_location="above")
    p2.add_layout(Legend(), "right")

    # Create a line for each category
    # Ensure event_date is datetime type
    dataframe[event_date] = pd.to_datetime(dataframe[event_date])

    unique_categories = dataframe[category].unique()
    for id, cat_name in enumerate(unique_categories):
        df_filtered = dataframe[dataframe[category] == cat_name].sort_values(by=event_date).reset_index(drop=True)

        p2.line(
            x=df_filtered[event_date],
            y=df_filtered[measure],
            line_width=2,
            line_color=color_palette[id % len(color_palette)],  # Cycle colors
            legend_label=str(cat_name), # Ensure legend label is a string
        )

    # Configure legend
    p2.legend.click_policy = "hide"
    p2.legend.location = "top_left" # Often better for time series to avoid obscuring data

    if subtitle is not None:
        p2.title.text = subtitle # Assign subtitle to the plot's title property
        p2.title.align = "center" # Center the subtitle

    # Create title figure (for the main overall title)
    title_fig = figure(
        title=title,
        toolbar_location=None,
        width=plot_width,
        height=40,
    )
    title_fig.title.align = "left"
    title_fig.title.text_font_size = "14pt"
    title_fig.border_fill_alpha = 0
    title_fig.outline_line_width = 0
    # Hide axes and tools for the title figure
    title_fig.xaxis.visible = False
    title_fig.yaxis.visible = False
    title_fig.xgrid.visible = False
    title_fig.ygrid.visible = False

    # Create subtitle figure (for the source information)
    sub_title_fig = figure( # Renamed to avoid conflict with `sub_title` variable
        title=source,
        toolbar_location=None,
        width=plot_width,
        height=40,
    )
    sub_title_fig.title.align = "left"
    sub_title_fig.title.text_font_size = "10pt"
    sub_title_fig.title.text_font_style = "normal"
    sub_title_fig.border_fill_alpha = 0
    sub_title_fig.outline_line_width = 0
    # Hide axes and tools for the source figure
    sub_title_fig.xaxis.visible = False
    sub_title_fig.yaxis.visible = False
    sub_title_fig.xgrid.visible = False
    sub_title_fig.ygrid.visible = False


    # Add event markers if provided
    if events_dict:
        used_y_positions = []

        for event_date_value, label_text in events_dict.items():
            # Add vertical line marker
            span = Span(
                location=pd.to_datetime(event_date_value), # Ensure location is datetime
                dimension="height",
                line_color='#C6C6C6',
                line_width=2,
                line_dash=(4, 4)
            )
            p2.renderers.append(span)

            # Determine label position to avoid overlap
            # Calculate y-range of the plot to scale label positions
            y_min, y_max = p2.y_range.start, p2.y_range.end
            y_range_span = y_max - y_min
            
            # Start label placement near the top
            base_y_position = y_max * 0.9 # 90% of max y-value
            label_offset = y_range_span / 20 # Small offset for each label

            # Adjust y_position to avoid overlap
            y_position = base_y_position
            while any(abs(y_position - used_y) < label_offset for used_y in used_y_positions):
                y_position -= label_offset
                if y_position < y_min: # Prevent labels from going off-plot downwards
                    y_position = base_y_position # Reset and try placing upwards or just break
                    break # Or handle more gracefully

            used_y_positions.append(y_position)

            # Add event label
            event_label = Label(
                x=pd.to_datetime(event_date_value), # Ensure x is datetime
                y=y_position,
                text=label_text,
                text_color="black",
                text_font_size="10pt",
                background_fill_color="grey",
                background_fill_alpha=0.2,
                x_offset=5, # Offset from the line
                y_offset=5,
                x_units='data', # Important for datetime x-axis
                y_units='data'
            )
            p2.add_layout(event_label)

    # Combine into a single layout
    layout = column(title_fig, p2, sub_title_fig) # Used sub_title_fig

    return layout

def create_regional_pollution_tabs(
    data_df: pd.DataFrame,
    temporal_granularity: str = 'Annual', # e.g., 'Annual', 'Monthly'
    date_column: str = 'date',
    category_column: str = 'ADM1_ES',
):
    """
    Generates and displays Bokeh tabs for regional air pollution data.

    The tabs include:
    1. Absolute Regional Air Pollution (e.g., NO2_mean)
    2. Percentage Change compared to 2019
    3. Percentage Change compared to Previous Year

    Parameters:
    -----------
    data_df : pd.DataFrame
        The input DataFrame containing the air pollution data.
        Expected columns: 'date', 'NO2_mean', 'percent_change_NO2_mean_2019',
        'percent_change_NO2_mean_PY', and the category_column (e.g., 'ADM1_ES').
    temporal_granularity : str, optional
        A string indicating the temporal resolution (e.g., 'Annual', 'Monthly').
        Used in the tab titles. Defaults to 'Annual'.
    date_column : str, optional
        The name of the column in data_df that contains date information. Defaults to 'date'.
    category_column : str, optional
        The name of the column in data_df that contains region identifiers. Defaults to 'ADM1_ES'.
    visuals_module : module
        The module containing the `get_line_plot` function (e.g., your `visuals.py` module).

    Returns:
    --------
    None
        Displays the Bokeh tabs directly.
    """
    # Silence Bokeh warnings about missing renderers (if you prefer)
    bokeh.core.validation.silence(MISSING_RENDERERS, True)
    bokeh.core.validation.silence(EMPTY_LAYOUT, True) # Added for completeness if this also occurs

    output_notebook() # Ensure this is called if running in a notebook environment

    # Ensure the date column is in datetime format and extract year
    data_df[date_column] = pd.to_datetime(data_df[date_column])
    data_df['year'] = data_df[date_column].dt.year # This column might be useful for internal logic or get_line_plot

    tabs = []
    # Dynamic titles based on temporal_granularity
    labels = [
        f'{temporal_granularity} Regional Air Pollution',
        '% Change in Regional Air Pollution compared to 2019',
        '% Change in Regional Air Pollution Compared to Previous Year'
    ]
    measures = ['NO2_mean', 'percent_change_NO2_mean_2019', 'percent_change_NO2_mean_PY']

    for idx, measure in enumerate(measures):
        # Get the line plot using the provided visuals_module
        p = get_line_plot(
            data_df,
            labels[idx],
            "Source: Sentinel 5-P extracted from Google Earth Engine",
            subtitle="", # Subtitle is empty in your usage
            category=category_column,
            measure=measure,
            event_date=date_column
        )

        tab = TabPanel(child=p, title=measure)
        tabs.append(tab)

    # Show the plot with tabs
    bokeh_tabs_display = Tabs(tabs=tabs, sizing_mode="scale_both")
    show(bokeh_tabs_display, warn_on_missing_glyphs=False)

def plot_regional_timeseries(df, variable_column, percent_change_column=None, time_column='event_date', 
                            region_column='ADM1_EN', geometry_df=None, geometry_region_column='ADM1_EN', 
                            years=None, exclude_years=None, pc_change=False, 
                            cmap='viridis', figsize=(16, 4), title=None,
                            highlight_regions=None, highlight_color='red', highlight_linewidth=2):
    """
    Plot regional values over time with options for percent change visualization.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame containing time series data by region
    variable_column : str
        Name of the column containing values to plot (e.g., 'NO2')
    percent_change_column : str
        Name of existing column containing percent change values (e.g., 'percent_change_NO2_PY')
    time_column : str
        Name of the column containing date information (default: 'event_date')
    region_column : str
        Name of the column containing region identifiers (default: 'ADM1_EN')
    geometry_df : GeoDataFrame
        GeoDataFrame containing region boundaries (default: None)
    geometry_region_column : str
        Name of region column in geometry_df (default: 'ADM1_EN')
    years : list
        List of years to include in plot (default: None, will use all years)
    exclude_years : list
        List of years to exclude from plot (default: None)
    pc_change : bool
        Whether to plot percent change instead of absolute values (default: False)
    cmap : str or matplotlib colormap
        Matplotlib colormap to use (default: 'viridis')
    figsize : tuple
        Figure size (width, height) in inches (default: (16, 4))
    title : str
        Custom title for the chart (default: None, will use auto-generated title)
    highlight_regions : list
        List of region names to highlight with a boundary (default: None)
    highlight_color : str
        Color to use for highlighted region boundaries (default: 'red')
    highlight_linewidth : int
        Line width for highlighted region boundaries (default: 2)
        
    Returns:
    --------
    fig : matplotlib Figure
        The figure object containing the plots
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import pandas as pd
    import numpy as np
    
    # Ensure the date column is a datetime
    df[time_column] = pd.to_datetime(df[time_column])
    
    # Determine years to plot if not specified
    if years is None:
        years = sorted(df[time_column].dt.year.unique())
    
    # Exclude specified years
    if exclude_years:
        years = [y for y in years if y not in exclude_years]
    
    # Create a figure with subplots (one for each year plus one for the colorbar)
    fig, ax = plt.subplots(1, len(years) + 1, figsize=figsize, 
                          gridspec_kw={'width_ratios': [1] * len(years) + [0.3]})
    
    # Flatten the axes array for easier indexing
    ax = ax.flatten()
    
    # Setup custom colormap for percent change if specified
    if pc_change:
        colors = ["#754493", "#A873C4", "#D1AEE3", "#EFEFEF", "#98CBCC", "#4EA2AC", "#24768E"]
        custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
        cmap = custom_cmap
    
    # Determine which column to plot based on whether we're plotting percent change
    plot_column = percent_change_column if pc_change and percent_change_column else variable_column
    
    # Get min and max values for consistent color scaling
    if pc_change and percent_change_column:
        
        vmin = df[df[time_column].dt.year.isin(years)][percent_change_column].min()
        #print(vmin)
        vmax = df[df[time_column].dt.year.isin(years)][percent_change_column].max()
        
        # For percent change, make sure the colormap is centered at 0
        # if abs(vmin) > abs(vmax):
        #     vmax = abs(vmin)
        # else:
        #     vmin = -abs(vmax)
    else:
        vmin = df[df[time_column].dt.year.isin(years)][variable_column].min()
        vmax = df[df[time_column].dt.year.isin(years)][variable_column].max()
    
    # Loop through each year and create a plot
    for i, year in enumerate(years):
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31'
        
        # Filter data for the current year
        year_data = df[(df[time_column] >= start_date) & (df[time_column] <= end_date)]
        
        # For percent change, we need to ensure the column exists
        if pc_change and percent_change_column not in year_data.columns:
            ax[i].text(0.5, 0.5, f"Missing column: {percent_change_column}",
                     ha='center', va='center', transform=ax[i].transAxes)
            continue
            
        # Select relevant columns
        if pc_change:
            year_data = year_data[[region_column, percent_change_column]]
        else:
            year_data = year_data[[region_column, variable_column]]
        
        # Ensure the values are numeric
        col_to_plot = percent_change_column if pc_change else variable_column
        year_data[col_to_plot] = year_data[col_to_plot].astype('float64')
        
        # Merge with geometry data and plot
        if geometry_df is not None:
            gdf = geometry_df.merge(year_data, left_on=geometry_region_column, 
                                   right_on=region_column, how='left')
            gdf.plot(column=col_to_plot, ax=ax[i], cmap=cmap, vmin=vmin, vmax=vmax)
            
            # Add highlighting for selected regions
            if highlight_regions is not None:
                # Create a subset of the GeoDataFrame with only the highlighted regions
                highlight_gdf = gdf[gdf[geometry_region_column].isin(highlight_regions)]
                if not highlight_gdf.empty:
                    # Plot the highlighted regions with a distinct boundary
                    highlight_gdf.boundary.plot(ax=ax[i], color=highlight_color, 
                                               linewidth=highlight_linewidth, zorder=2)
                    
                    # Add a legend for the highlighted regions
                    if i == 0:  # Only add legend to the first subplot
                        from matplotlib.lines import Line2D
                        custom_line = Line2D([0], [0], color=highlight_color, lw=highlight_linewidth)
                        ax[i].legend([custom_line], ['Highlighted Regions'], 
                                    loc='upper left', fontsize=8)
        else:
            ax[i].text(0.5, 0.5, "Geometry data required for maps", 
                     ha='center', va='center', transform=ax[i].transAxes)
        
        # Set the title for each subplot
        ax[i].set_title(f'{year}')
        
        # Clean up the plot
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        for spine in ax[i].spines.values():
            spine.set_visible(False)
        ax[i].grid(False)
    
    # Add the colorbar (legend) in the last subplot
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Empty array for the colorbar
    
    # Create the colorbar
    cbar = fig.colorbar(sm, cax=ax[-1])
    
    # Format colorbar with percentage labels for percent change
    if pc_change:
        ticks = np.linspace(vmin, vmax, num=5)  # Define tick positions
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.1f}%" for t in ticks])  # Format as percentages
        cbar.set_label(f"{variable_column} % Change", fontsize=12, fontweight="bold")
    else:
        cbar.set_label(variable_column, fontsize=12, fontweight="bold")
    
    # Set the main title
    if title is not None:
        plt_title = title
    else:
        if pc_change:
            plt_title = f'Percentage Change in Annual {variable_column} Compared to Previous Year per Region'
        else:
            plt_title = f'Regional Annual {variable_column} Levels'
    
    plt.suptitle(plt_title, fontsize=16, fontweight='bold')
    
    # Add a footer text
    plt.figtext(0.01, 0.01, 
               f"Source: NO2 values from Sentinel-5P. Admin boundaries from HdX. World Bank calculations.", 
               ha="left", fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig