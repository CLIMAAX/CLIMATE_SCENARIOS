# %%
import ipywidgets as widgets
from ipywidgets import interact
from IPython.display import clear_output
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def plot_widget():
    # Define variables and their full names
    atmos_variables = [
        'tas', 'tasmin', 'TNn', 'fd', 'hdd',
        'tasmax', 'TXx', 'tx35', 'TX35bc', 'tx40', 'TX40bc', 'cdd',
        'pr', 'Rx1day', 'Rx5day', 'ds', 'spi6',
        'prsn', 'sfcWind'
    ]

    # Mapping of variable codes to their full names
    variable_full_names = {
        'tas': 'Mean Temperature',
        'tasmin': 'Minimum Temperature',
        'TNn': 'Minimum of Minimum Temperature',
        'fd': 'Frost Days',
        'hdd': 'Heating Degree Days',
        'tasmax': 'Maximum Temperature',
        'TXx': 'Maximum of Maximum Temperature',
        'tx35': 'Days with Temperature > 35°C',
        'TX35bc': 'Days with Temperature > 35°C (Bias Corrected)',
        'tx40': 'Days with Temperature > 40°C',
        'TX40bc': 'Days with Temperature > 40°C (Bias Corrected)',
        'cdd': 'Cooling Degree Days',
        'pr': 'Total Precipitation',
        'Rx1day': 'Maximum 1-Day Precipitation',
        'Rx5day': 'Maximum 5-Day Precipitation',
        'ds': 'Consecutive Dry Days',
        'spi6': 'Standardized Precipitation Index (6 months)',
        'prsn': 'Total Snowfall',
        'sfcWind': 'Surface Wind Speed'
    }

    # Mapping of variable codes to their units
    variable_units = {
        'tas': '°C',
        'tasmin': '°C',
        'tasmax': '°C',
        'TNn': '°C',
        'TXx': '°C',
        'pr': '%',
        'Rx1day': '%',
        'Rx5day': '%',
        'spi6': '%',
        'sfcWind': '%',
        'prsn': 'mm/day',
        'cdd': 'Degree Days',
        'hdd': 'Degree Days'
    }

    # Define scenarios and their corresponding colors for each dataset
    scenarios_and_colors = {
        'CMIP6': {
            'scenarios': ['ssp126', 'ssp245', 'ssp370', 'ssp585'],
            'colors': {
                'ssp126': '#402575',  # Deep blue
                'ssp245': '#91ADD3',  # Light blue
                'ssp370': '#F19D6F',  # Orange
                'ssp585': '#E64139'   # Red
            }
        },
        'CORDEX': {
            'scenarios': ['rcp26', 'rcp45', 'rcp85'],
            'colors': {
                'rcp26': '#402575',  # Deep blue
                'rcp45': '#91ADD3',  # Light blue
                'rcp85': '#E64139'   # Red
            }
        }
    }

    # Define the available regions and variables
    regions = ['NEU', 'MED', 'WCE', 'EEU']
    variables = [(variable_full_names[var], var) for var in atmos_variables]
    time_filters = ['year', 'DecFeb', 'MarMay', 'JunAug', 'SepNov']
    timeframes = [2021, 2041, 2081]

    # Create the dropdowns for user interaction
    region_dropdown = widgets.Dropdown(
        options=regions,
        value='NEU',
        description='Region:',
    )

    data_set_dropdown = widgets.Dropdown(
        options=['CMIP6', 'CORDEX'],
        value='CMIP6',
        description='Dataset:',
    )

    variable_dropdown = widgets.Dropdown(
        options=variables,
        value='tas',
        description='Variable:',
    )

    time_filter_dropdown = widgets.Dropdown(
        options=time_filters,
        value='year',
        description='Time Filter:',
    )

    timeframe_dropdown = widgets.Dropdown(
        options=timeframes,
        value=2041,
        description='Timeframe:',
    )

    # Flag to check if it's the first click
    first_click = True

    # Function to update the plot based on user selection
    def update_plot(region, dataset, variable, time_filter, timeframe):
        nonlocal first_click
        clear_output(wait=True)  # Clear previous output
        ref_time = 'preIndustrial'
        landmask = 'True'
        variable_full_name = variable_full_names.get(variable, variable)
        unit = variable_units.get(variable, 'days')

        # Load the data
        try:
            data = pd.read_csv(
                f'data/{region}/{region}_{dataset}_{ref_time}_landmask={landmask}.csv'
            )
        except FileNotFoundError:
            print("Data file not found.")
            return

        # Process the data
        data = data.set_index(['variable', 'model', 'date', 'scenario', 'season'])
        data = data.loc[(variable, slice(None), timeframe, slice(None), time_filter)]
        data_restructured = data.reset_index()
        data_restructured = data_restructured.set_index('model')
        data_restructured = data_restructured.rename(columns={'scenario': 'ssp', 'value': 'impact'})

        # Check if the data is empty
        if data_restructured.empty:
            # Create an empty figure with a message
            fig = go.Figure()
            fig.update_layout(
                height=500,
                margin=dict(l=20, r=20, t=50, b=50),
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                title=dict(
                    text=f'Projected changes {dataset} in {region} for {variable_full_name} during {time_filter} between {timeframe}-{int(timeframe) + 19}',
                    x=0.5,
                    xanchor='center'
                )
            )
            fig.add_annotation(
                text="Data set not available, try another one",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=25, color="black"),
                align="center",
                valign="middle",
            )
            fig.show()
            return

        # Get multi-model mean (mmm)
        mmm = data_restructured.loc[data_restructured.index == 'mmm']

        # Assign SSP impacts based on scenarios
        data_restructured['ssp_impact'] = np.nan
        scenarios = scenarios_and_colors[dataset]['scenarios']
        colors = scenarios_and_colors[dataset]['colors']

        for scenario in scenarios:
            if scenario in mmm['ssp'].values:
                mmm_value = mmm[mmm['ssp'] == scenario]['impact'].values[0]
                data_restructured.loc[data_restructured['ssp'] == scenario, 'ssp_impact'] = mmm_value

        # Prepare data for plotting
        data_restructured.reset_index(inplace=True)
        df = data_restructured[['impact', 'ssp', 'ssp_impact', 'model']]

        # Determine x-axis range
        x_min = df['impact'].min()
        x_max = df['impact'].max()
        range_extension = 0.05 * (x_max - x_min)
        x_range = [x_min - range_extension, x_max + range_extension]

        # Mapping of y-values to numerical positions
        y_mapping = {'Model': 0, 'SSP': 1, 'SSP Extension': 2}

        # Initialize figure
        fig = go.Figure()

        # Plot lines for each model and SSP
        for _, entry in df.iterrows():
            ssp = entry['ssp']
            color = colors.get(ssp, 'black')
            model_name = entry['model']

            if model_name == 'mmm':
                x_vals = [entry['ssp_impact'], entry['ssp_impact']]
                y_vals = [y_mapping['SSP'], y_mapping['SSP Extension']]
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode='lines',
                        line=dict(color=color, width=2),
                        showlegend=False,
                        hoverinfo='text',
                        text=f"Model: {model_name}<br>SSP: {ssp}<br>Impact: {entry['ssp_impact']:.2f} {unit}"
                    )
                )
                continue

            x_vals = [entry['impact'], entry['ssp_impact']]
            y_vals = [y_mapping['Model'], y_mapping['SSP']]
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines',
                    line=dict(color=color, width=2),
                    showlegend=False,
                    hoverinfo='text',
                    text=f"Model: {model_name}<br>SSP: {ssp}<br>Change: {entry['impact']:.2f} {unit}"
                )
            )

        # Set x-axis properties
        xaxis_title = f'Change in {variable_full_name} [{unit}]'
        fig.update_xaxes(
            title_text=xaxis_title,
            range=x_range,
            tickmode='auto',
            showgrid=False,
            zeroline=False,
            ticks='outside',
            ticklen=3,
            tickwidth=1,
            tickcolor='black',
            showline=True,
            linecolor='black',
            linewidth=1
        )

        # Set y-axis properties
        fig.update_yaxes(
            tickmode='array',
            tickvals=[y_mapping['Model'], y_mapping['SSP'], y_mapping['SSP Extension']],
            ticktext=['Model', 'SSP', ''],
            showgrid=False,
            zeroline=False
        )

        # Adjust layout properties
        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=50, b=50),
            plot_bgcolor='white',
            paper_bgcolor='white',
            title=dict(
                text=f'Projected changes {dataset} in {region} for {variable_full_name} during {time_filter} between {timeframe}-{int(timeframe) + 19}',
                x=0.5,
                xanchor='center'
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.05,
                xanchor="center",
                x=0.5
            )
        )

        # Add a manual legend for SSPs
        for ssp, color in colors.items():
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode='lines',
                    line=dict(color=color, width=4),
                    name=ssp
                )
            )

        # Show the figure
        if first_click == False:
            fig.show()
        else:
            first_click = False

    # Create the interactive widget
    interact(
        update_plot,
        region=region_dropdown,
        dataset=data_set_dropdown,
        variable=variable_dropdown,
        time_filter=time_filter_dropdown,
        timeframe=timeframe_dropdown
    )

# %% [markdown]
# To guide region slection here is a classification of what areas are in Europe
# 
# <img src="regions_map.png" alt="EU regions" width="400">
# 
# Image taken from [IPCC Atlas](https://doi.org/10.1017/9781009157896.021)

# %% [markdown]
# 


