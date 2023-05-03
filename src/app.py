#############
## Imports ##
#############

import io
import time
import base64
import collections
from itertools import combinations
from functools import lru_cache
from typing import Union,List

import pandas as pd
import numpy as np

from zipfile import BadZipFile

import dash
from dash import html, Dash,  dcc, Input, Output, State
from dash.dash_table import DataTable
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash.long_callback import DiskcacheLongCallbackManager
import plotly.graph_objs as go
import plotly.express as px

from sklearn.cluster import KMeans

import diskcache

from here_location_services import LS



############
## Config ##
############

# Color options
color_options = {
    'demand': 'red',
    'supply': 'yellow',
    'flow': 'black',
    'cog': 'blue',
    'candidate': 'black',
    'other': 'gray'
}

here_maps_app_id = "dQz8c3OpaYY2qDnxgqvR"
here_maps_api_key = "81XB7NBfyqEHfFE8lEVDGywwJntlRNlzBUV7jXUBXh0"
mapbox_access_token = 'pk.eyJ1Ijoic2NhbGZsIiwiYSI6ImNrZXY0eWxwcjA5cGwyc3FvZDQ4MXRsZW8ifQ.0Kk19McNcx7a8TdcfzlNRA'

cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

app = dash.Dash(
    __name__,
    url_base_pathname='/lfl-sca-cog/',
    title = "Center of Gravity",
    external_stylesheets=[dbc.themes.BOOTSTRAP,"assets/css/styles.css"],
    background_callback_manager=long_callback_manager
)
server=app.server
ls = LS(api_key=here_maps_api_key)  

###############
## Functions ##
###############

def retry(times, exceptions):

    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    print(
                        'Exception thrown when attempting to run %s, attempt '
                        '%d of %d' % (func, attempt, times)
                    )
                    attempt += 1
                    time.sleep(attempt)
            return func(*args, **kwargs)
        return newfn
    return decorator

@cache.memoize()
@retry(times=100,exceptions=(Exception))
def geocode_location(location_addres:str):
    longitude, latitude, country, state, city = None, None, None, None, None
    geo = ls.geocode(query=str(location_addres)) 
    geo = geo.to_geojson()
    try:
        
        coordinates = geo.get('features')[0].get('geometry',{}).get('coordinates')
        longitude = coordinates[0]
        latitude = coordinates[1]
        _address = geo.get('features')[0].get('properties',{}).get('address',{})
        country = _address.get('countryName')
        state = _address.get('state')
        city = _address.get('city')

    except Exception as e:
        return longitude, latitude, country, state, city
    return longitude, latitude, country, state, city

def render_table(dataframe, table_id, editable=False, page_size=250, container_width='65%', style_data={'whiteSpace': 'normal','height': 'auto','backgroundColor': 'rgb(100, 100, 100)','color': 'white'}, style_cell={'height': 'auto','minWidth': '180px', 'width': '180px', 'maxWidth': '180px', 'whiteSpace': 'normal' }, style_table={'height': '26rem', 'overflowY': 'auto','overflowX': 'scroll','padding':'1rem','max-width':'65rem'}):
    return dbc.Container(
        children=DataTable(
            id=table_id,
            columns=[{'name': i, 'id': i} for i in dataframe.columns],
            editable=editable,
            export_format='xlsx',
            export_headers='display',
            data=dataframe.to_dict('records'),
            fixed_rows={'headers': True},
            page_size=page_size,
            style_data=style_data,
            style_data_conditional=[
                {
                    'if': {
                        'column_id': f"{_col}",
                        'filter_query': '{final_lat} eq 0'
                    },
                    'backgroundColor': 'red',
                } for _col in dataframe.columns
            ],
            style_cell=style_cell,
            style_table=style_table,
            css=[{'selector':'.export','rule':'position:absolute;left:-15px;bottom:-30px'}],
        ),
        style={'width':container_width}
    )
     

def getBoundsZoomLevel(bounds, mapDim):
    """source: https://stackoverflow.com/questions/6048975/google-maps-v3-how-to-calculate-the-zoom-level-for-a-given-bounds
    bounds: list of ne and sw lat/lon
    mapDim: dictionary with image size in pixels
    returns: zoom level to fit bounds in the visible area
    """
    ne_lat = bounds[0]
    ne_long = bounds[1]
    sw_lat = bounds[2]
    sw_long = bounds[3]

    scale = 2 # adjustment to reflect MapBox base tiles are 512x512 vs. Google's 256x256
    WORLD_DIM = {'height': 256 * scale, 'width': 256 * scale}
    ZOOM_MAX = 18

    def latRad(lat):
        sin = np.sin(lat * np.pi / 180)
        radX2 = np.log((1 + sin) / (1 - sin)) / 2
        return max(min(radX2, np.pi), -np.pi) / 2

    def zoom(mapPx, worldPx, fraction):
        return np.floor(np.log(mapPx / worldPx / fraction) / np.log(2))

    latFraction = (latRad(ne_lat) - latRad(sw_lat)) / np.pi

    latFraction = latFraction if latFraction != 0 else 1

    lngDiff = ne_long - sw_long
    lngFraction = ((lngDiff + 360) if lngDiff < 0 else lngDiff) / 360

    lngFraction = lngFraction if lngFraction != 0 else 1

    latZoom = zoom(mapDim['height'], WORLD_DIM['height'], latFraction)
    lngZoom = zoom(mapDim['width'], WORLD_DIM['width'], lngFraction)

    return min(latZoom, lngZoom, ZOOM_MAX)

def loc_type_mult(x, ratio):
    """A function to get the volume multiplier based on the location type and the IB-OB ratio.
    x: The location type
    """
    if x.lower() == 'supply':
        # No need to divide since we are already multiplying the demand
        return 1
    elif x.lower() == 'demand':
        # Only apply multiplier to demand
        return ratio
    else:
        # If neither supply nor demand, remove entirely
        return 0
    
############
## Layout ##
############

app.layout = dbc.Container(
    children=[
        html.Button(id='help_button',children=[html.Img(src="assets/icons/question.png", height=30,title="Help")],style={'position':'fixed','left':'95%','top':'5%','border':'none','background': 'none','opacity':'60%'}), #TO-DO
        dbc.Col(
            children=[
                dbc.Spinner(id="common_spinner",color="primary",spinner_style={"display":"none"}),
                dbc.Alert("temp_alert",id="auto_alert",is_open=False,duration=3000,color="danger"),
                dcc.Store(id='settings_hash',storage_type='memory'),
                dcc.Store(id='output_hash',storage_type='memory'),
                dbc.Container(
                    children=[html.H4('Feature Will Be Added Soon',style={'opacity':'50%'})],
                    id='help_tab',
                    style={'display':'none'}
                ),
    

                dbc.Row(html.H1("Center of Gravity"),style={'text-align':'center','border-bottom':'solid black 0.15rem'}), # Title
                dbc.Row(
                    children=[
                        dbc.Col(children=[dbc.Container(
                            children=[
                                dbc.Row(children=[html.H3("Settings",style={"padding-top":"1rem"})] , style={'text-align':'center'},align="start"),
                                dbc.Row(children=[dbc.Form(children=[
                                    
                                    dcc.Upload(id='cog_input_excel', children=html.Div([ 'Click Here To Upload Files '], id='upload_file_text')),
                                    dbc.FormText("Upload COG Input Excel", color="secondary"),
                                    html.Br(),html.Br(),
                                    dbc.Input(type="number", min=0, max=1000, step=1, value=1, id="no_of_cog",placeholder="No. of COG"),
                                    dbc.FormText("Enter the Number of Center of Gravities", color="secondary"),
                                    html.Br(),html.Br(),
                                    dbc.Input(type="number", min=0, max=1000, step=0.001, value=1, id="ibob_ratio",placeholder="Inbound-Outbound Ratio"),
                                    dbc.FormText("Enter the Inbound-Outbound Ratio", color="secondary"),
                                    dbc.Tooltip("The IB-OB ratio allows analysts to adjust the relative transport costs of IB and OB lanes. A ratio of 2 means that OB transport is 2 times as expensive as IB transport. A ratio of 0.5 means that OB transport is 0.5 times as expensive as IB transport.",target="ibob_ratio",),
                                    html.Br(),html.Br(),
                                    dbc.Checklist(
                                        options=[
                                            {"label": "Use Actual Road Distances", "value": 1, "disabled":True},
                                            {"label": "Avoid Water Bodies", "value": 2, "disabled":True},
                                        ],value=[],id="switches_input",switch=True,
                                    ),

                                ])],align="center",style={'margin-top':'1rem','margin-bottom':'1rem','text-align':'center'}),
                                html.Br(),
                                dbc.Row(children=[dbc.Button("Calculate COG",id="calculate_button")] , style={'text-align':'center','padding':'1rem'},align="end")
                            ]
                        
                        )],style={'border':'solid black 0.15rem','border-radius':'0.5rem'},width={'size':3,'offset':0,'order':'first'},id='input_section'),
                        
                        dbc.Col(children=[dbc.Container(
                            children=[
                                dbc.Row(children=[html.H3("Output",style={"padding-top":"1rem"})] , style={'text-align':'center'},align="start",justify="between"),
                                dbc.Row(children=[
                                    html.Div(id="input_data_table_panel",style={"display":"none"}),
                                    html.Div(id="input_data_heatmap_panel",style={"display":"none"}),
                                    html.Div(id="input_data_summary_panel",style={"display":"none"}),
                                    html.Div(id="cog_map_panel",style={"display":"none"}),
                                    html.Div(id="cog_table_panel",style={"display":"none"}),

                                    html.Div(id="starter_image_panel",children=[html.Img(src="assets/images/continents-world-background_53876-116652.png", style={'width':'100%','height':'auto'})]),
                                ],justify="between"),
                                dbc.Row(children=[
                                    dbc.RadioItems(
                                        id="output_navigator",
                                        className="btn-group",
                                        inputClassName="btn-check",
                                        labelClassName="btn btn-outline-primary",
                                        labelCheckedClassName="active", 
                                        options=[
                                            {"label": [html.Img(src="assets/icons/icons8-summary-list-50.png", height=30,title="Input Data Table")], "value": 1},
                                            {"label": [html.Img(src="assets/icons/icons8-heat-map-50.png", height=30,title="Input Data Heat Map")], "value": 2},
                                            {"label": [html.Img(src="assets/icons/icons8-financial-analytics-80.png", height=30,title="Input Data Summary")], "value": 3, "disabled":True},
                                            {"label": [html.Img(src="assets/icons/icons8-center-of-gravity-50.png", height=30,title="Center of Gravity Map")], "value": 4},
                                            {"label": [html.Img(src="assets/icons/icons8-data-sheet-50.png", height=30,title="Center of Graity Table")], "value": 5},
                                            
                                        ],
                                        value=4,
                                        # style={"visibility":"hidden"},
                                    ),
                                ],align="end",justify="between"  ),

                            ] , style={"display":"flex","flex-direction":"column","align-items":"center",}                          
                        ),],style={'border':'solid black 0.15rem','border-radius':'0.5rem',},width={'size':8,'offset':0,'order':'last'}, id='output_section')
                    ],
                    style={'margin-top':'1rem'},
                    justify="evenly"
                ),
            ],
            # style={''}
        ),  
    ],
    className="p-3"
)

###############
## Callbacks ##
###############

@app.callback(
    Output(component_id='settings_hash', component_property='data'),
    Input(component_id='cog_input_excel', component_property='contents'),
    Input(component_id='no_of_cog', component_property='value'),
    Input(component_id='ibob_ratio', component_property='value'),
    Input(component_id='switches_input', component_property='value'),
    prevent_initial_call=True
)
def generate_settings_hash(cog_input_excel:Union[str,None],no_of_cog:Union[int,None],ibob_ratio:Union[float,None],switches_input:Union[List[int],None]):
    # Validate and Clean Data
    no_of_cog = 0 if not no_of_cog else no_of_cog
    ibob_ratio = 1 if not ibob_ratio else ibob_ratio
    switches_input = [] if not switches_input else switches_input
    # Consolidate Data to form Hash String
    hash_string = f"{cog_input_excel}{no_of_cog}{ibob_ratio}{switches_input}"
    # Calculate Hashed String
    hashed_string = hash(hash_string)
    return hashed_string

@app.long_callback(
    output=[
    
        Output("input_data_table_panel","children"),
        Output("input_data_heatmap_panel","children"),
        Output("input_data_summary_panel","children"),
        Output("cog_map_panel","children"),
        Output("cog_table_panel","children"),

        Output("output_hash", "data"),
        Output("auto_alert","is_open"),
        Output("auto_alert","children"),
    ],
    inputs=[
        Input("calculate_button", "n_clicks"),
        State(component_id='cog_input_excel', component_property='contents'),
        State(component_id='cog_input_excel', component_property='filename'),
        State(component_id='no_of_cog', component_property='value'),
        State(component_id='ibob_ratio', component_property='value'),
        State(component_id='switches_input', component_property='value'),
        State(component_id='settings_hash', component_property='data')
    ],
    running=[
        (Output("calculate_button", "disabled"), True, False),
        (Output("common_spinner", "style"), {"width": "3rem", "height": "3rem",'position':'fixed','left':'60%','top':'45%',"display":"flex","z-index":"1001"}, {"display":"none"}),
    ],
    prevent_initial_call=True
)
def generate_cog_analysis(n_clicks:int,cog_input_excel:Union[str,None],filename:str,no_of_cog:Union[int,None],ibob_ratio:Union[float,None],switches_input:Union[List[int],None],settings_hash:str):

    # Validate Data

    if not cog_input_excel:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, settings_hash, True, "Please Upload a File"
    if not no_of_cog:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, settings_hash, True, "Please Enter The Number of COG's"
    
    if not filename.endswith('.xlsx'):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, settings_hash, True, "Please Upload a .xlsx File Following the Template"

    # Convert File Data to DF
    _ , content_string = cog_input_excel.split(',')
    cog_input_excel = base64.b64decode(content_string) 

    try:
        df = pd.read_excel(io.BytesIO(cog_input_excel),engine='openpyxl',sheet_name="cog_data")
    except BadZipFile as e:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, settings_hash, True, "Unable to Read The File Uploaded"
    except ValueError as e:
        if 'cog_data' in str(e).lower():
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, settings_hash, True, "Unable to Find Sheet Named cog_data"
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, settings_hash, True, f"Unexpected Error : {str(e)}"
    except Exception as e:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, settings_hash, True, f"Unexpected Error : {str(e)}"

    expected_columns = ['Latitude', 'Location Address', 'Location ID', 'Location Type', 'Longitude', 'Volume']
    if not collections.Counter(list(df.columns)) == collections.Counter(expected_columns):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, settings_hash, True, f"Excel Sheet doesn't Follow Template Format"

    df['Location Type']=df['Location Type'].map(str.capitalize)
    uq_location_type = df['Location Type'].unique()
    expected_location_types =  ["Demand","Supply","Candidate",]
    if not ( set(uq_location_type) <= set(expected_location_types)  )  :
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, settings_hash, True, f"Invalid Location Types"

    # Add geo_lat and geo_lon if Lat and Lon are not availible
    df['is_geo_code'] = df.apply(lambda x: False if (pd.notnull(x['Latitude']) and pd.notnull(x['Longitude'])) else True, axis=1)
    # df['final_loc'] = df.apply(lambda x: str(geocode_location(x['Location Address'])) if (x['is_geo_code'] == True) else f"({x['Longitude']},{x['Latitude']})", axis=1)
    # df['final_lat'] = df.apply(lambda x: x['final_loc'].split(',')[1].replace('(','').replace(')','').strip(), axis=1)
    # df['final_lon'] = df.apply(lambda x: x['final_loc'].split(',')[0].replace('(','').replace(')','').strip(), axis=1)
    df['final_loc'] = df.apply(lambda x: str(geocode_location(x['Location Address'])), axis=1)
    df['final_lat'] = df.apply(lambda x: x['Latitude'] if not (x['is_geo_code'] == True) else x['final_loc'].split(',')[1].replace('(','').replace(')','').strip(), axis=1)
    df['final_lon'] = df.apply(lambda x: x['Longitude'] if not (x['is_geo_code'] == True) else x['final_loc'].split(',')[0].replace('(','').replace(')','').strip(), axis=1)
    df['final_country'] = df.apply(lambda x: x['final_loc'].split(',')[2].replace('(','').replace(')','').strip(), axis=1)
    df['final_state'] = df.apply(lambda x: x['final_loc'].split(',')[3].replace('(','').replace(')','').strip(), axis=1)
    df['final_city'] = df.apply(lambda x: x['final_loc'].split(',')[4].replace('(','').replace(')','').strip(), axis=1)

    df['final_lat'].replace('None', 0, inplace=True)
    df['final_lon'].replace('None', 0, inplace=True)

    df["final_lat"] = pd.to_numeric(df["final_lat"])
    df["final_lon"] = pd.to_numeric(df["final_lon"])
    df['Volume'] = df['Volume'].apply(float)
    
    # Generate Input Table

    # uq_countries =  df['country'].unique()
    input_table = None
    
    # Generate Input Heat Map
    map_height = 350
    map_width = 450
    heatmap_radius = 25
    fig1 = go.Figure()

    # Demand sites
    fig1.add_trace(go.Scattermapbox(
        name='Demand',
        mode = 'markers',
        hovertemplate = df.loc[df['Location Type'].str.lower()=='demand', 'Location Address'],
        lon = df.loc[df['Location Type'].str.lower()=='demand', 'final_lon'],
        lat = df.loc[df['Location Type'].str.lower()=='demand', 'final_lat'],
        marker = {'size': 10,
                    'color': color_options['demand']}))

    # Supply sites
    fig1.add_trace(go.Scattermapbox(
        name='Supply',
        mode = 'markers',
        hovertemplate = df.loc[df['Location Type'].str.lower()=='supply', 'Location Address'],
        lon = df.loc[df['Location Type'].str.lower()=='supply', 'final_lon'],
        lat = df.loc[df['Location Type'].str.lower()=='supply', 'final_lat'],
        marker = {'size': 10,
                    'color': color_options['supply']}))
        
    # Heat map of volumes
    fig1.add_trace(go.Densitymapbox(
        lat=df['final_lat'],
        lon=df['final_lon'],
        z=df['Volume'],
        radius=heatmap_radius,
        showscale=False))

    fig1.update_layout(
        margin ={'l':0,'t':0,'b':0,'r':0},
        mapbox = {
            'accesstoken': mapbox_access_token,
            'center': {'lon': df['final_lon'].mean(),
                        'lat': df['final_lat'].mean()},
            'bearing': 0,
            #'style': 'stamen-terrain',
            'zoom': getBoundsZoomLevel([df['final_lat'].max(),
                                        df['final_lon'].max(),
                                        df['final_lat'].min(),
                                        df['final_lon'].min()],
                                        {'height': map_height,
                                        'width': map_width})})
    
    # Create Pie Chart
    _df_demand = df[df['Location Type'] == 'Demand']
    _df_supply = df[df['Location Type'] == 'Supply']
    fig2 = px.pie(_df_demand, values='Volume', names='Location Address', hole=.3)
    fig3 = px.pie(_df_supply, values='Volume', names='Location Address', hole=.3)

    # Calculate COG Map
    df['Calc_Vol'] = df['Location Type'].apply(str).apply(loc_type_mult, ratio=ibob_ratio) * df['Volume']

    # Use clustering if no candidate sites and use nearest neighbors if there are
    if 'Candidate' in uq_location_type or (df['Location Type'].str.lower()=='candidate').sum()<no_of_cog:
        print('Using K-means')
        # Fit K-means
        kmeans = KMeans(n_clusters=no_of_cog,
                        random_state=0).fit(df.loc[df['Calc_Vol']>0, ['final_lat',
                                                                            'final_lon']],
                                            sample_weight=list(df.loc[df['Calc_Vol']>0,
                                                                        'Calc_Vol']))

        # Get centers of gravity from K-means
        cogs = kmeans.cluster_centers_
        cogs = pd.DataFrame(cogs, columns=['final_lat',
                                            'final_lon'])

        # Get volume assigned to each cluster
        df['Cluster'] = kmeans.predict(df[['final_lat', 'final_lon']])
        cogs = cogs.join(df.groupby('Cluster')['Volume'].sum())
        cogs = cogs.join(df.rename(columns={'Volume': 'Count'}).groupby('Cluster')['Count'].size())

        # Include assigned COG coordinates in data by point
        df = df.join(cogs, on='Cluster', rsuffix='_COG')
    else:
        print('Using iteration')
        cands = df.loc[df['Location Type'].str.lower()=='candidate']
        locs = df.loc[df['Calc_Vol']>0]
        
        total_dist = np.inf
        best_cogs = []

        # Loop to find best combination of candidate sites
        for i in list(combinations(cands.index, no_of_cog)):
            temp_cands = cands.loc[list(i)]
            locs['Cluster'] = 0
            locs['Distance_COG'] = np.inf
            for i_l, r_l in locs.iterrows():
                for i_c, r_c in temp_cands.iterrows():
                    # Get distance
                    dist = (r_l['final_lat']-r_c['final_lat'])**2
                    dist += (r_l['final_lon']-r_c['final_lon'])**2
                    dist **= 0.5
                    # Save values if distance is shorter
                    if dist < locs.loc[i_l, 'Distance_COG']:
                        # Save distance
                        locs.loc[i_l, 'Distance_COG'] = dist
                        # Save index of nearest point
                        locs.loc[i_l, 'Cluster'] = i_c
            # Weight distance by volume
            locs['Weighted_Distance_COG'] = locs['Distance_COG'] * locs['Calc_Vol']
            # Save scenario if total weighted distance is smaller
            if locs['Weighted_Distance_COG'].sum() < total_dist:
                total_dist = locs['Weighted_Distance_COG'].sum()
                best_cogs = list(list(i))
                
        # Get centers of gravity
        cogs = cands.loc[best_cogs, ['final_lat',
                                        'final_lon']]
        # Reloop to get site assignment
        locs['Cluster'] = 0
        locs['Distance_COG'] = np.inf
        for i_l, r_l in locs.iterrows():
            for i_c, r_c in cogs.iterrows():
                # Get distance
                dist = (r_l['final_lat']-r_c['final_lat'])**2
                dist += (r_l['final_lon']-r_c['final_lon'])**2
                dist **= 0.5
                # Save values if distance is shorter
                if dist < locs.loc[i_l, 'Distance_COG']:
                    # Save distance
                    locs.loc[i_l, 'Distance_COG'] = dist
                    # Save index of nearest point
                    locs.loc[i_l, 'Cluster'] = i_c
                
        # Get volume assigned to each cog
        cogs = cogs.join(locs.groupby('Cluster')['Volume'].sum())
        cogs = cogs.join(locs.rename(columns={'Volume': 'Count'}).groupby('Cluster')['Count'].size())
    
        # Include assigned COG coordinates in data by point
        df = df.join(locs['Cluster'])
        df = df.join(cogs, on='Cluster', rsuffix='_COG')
    
    cogs['geocode_results'] = np.nan
    cogs['formatted_address'] = ''
    cogs['city'] = 'temp'

    # Initialize figure sites
    fig4 = go.Figure()

    # Build demand trace list
    lat_trace_d = {}
    lon_trace_d = {}
    for i in df.loc[df['Location Type'].str.lower()=='demand', 'Cluster'].dropna().unique():
        i = int(i)
        lat_trace_d[i] = []
        lon_trace_d[i] = []
        for index, row in df.loc[(df['Location Type'].str.lower()=='demand')&
                                    (df['Cluster']==i)].iterrows():
            lat_trace_d[i].append(row['final_lat_COG'])
            lat_trace_d[i].append(row['final_lat'])
            lon_trace_d[i].append(row['final_lon_COG'])
            lon_trace_d[i].append(row['final_lon'])

    # Build supply trace list
    lat_trace_s = {}
    lon_trace_s = {}
    for i in df.loc[df['Location Type'].str.lower()=='supply', 'Cluster'].dropna().unique():
        i = int(i)
        lat_trace_s[i] = []
        lon_trace_s[i] = []
        for index, row in df.loc[(df['Location Type'].str.lower()=='supply')&
                                    (df['Cluster']==i)].iterrows():
            lat_trace_s[i].append(row['final_lat_COG'])
            lat_trace_s[i].append(row['final_lat'])
            lon_trace_s[i].append(row['final_lon_COG'])
            lon_trace_s[i].append(row['final_lon'])

    # Demand sites and traces
    for i in lon_trace_d:
        fig4.add_trace(go.Scattermapbox(
            name='Demand',
            mode = 'lines+markers',
            lon = lon_trace_d[i],
            lat = lat_trace_d[i],
            marker = {'size': 10,
                        'color': color_options['demand']},
            showlegend=False))

    # Supply sites and traces
    for i in lon_trace_s:
        fig4.add_trace(go.Scattermapbox(
            name='Supply',
            mode = 'lines+markers',
            lon = lon_trace_s[i],
            lat = lat_trace_s[i],
            marker = {'size': 10,
                        'color': color_options['supply']},
            showlegend=False))

    # Demand sites
    fig4.add_trace(go.Scattermapbox(
        name='Demand',
        mode = 'markers',
        hovertemplate = df.loc[df['Location Type'].str.lower()=='demand', 'Location Address'],
        lon = df.loc[df['Location Type'].str.lower()=='demand', 'final_lon'],
        lat = df.loc[df['Location Type'].str.lower()=='demand', 'final_lat'],
        marker = {'size': 10,
                    'color': color_options['demand']}))

    # Supply sites
    fig4.add_trace(go.Scattermapbox(
        name='Supply',
        mode = 'markers',
        hovertemplate = df.loc[df['Location Type'].str.lower()=='supply', 'Location Address'],
        lon = df.loc[df['Location Type'].str.lower()=='supply', 'final_lon'],
        lat = df.loc[df['Location Type'].str.lower()=='supply', 'final_lat'],
        marker = {'size': 10,
                    'color': color_options['supply']}))

    # Centers of gravity
    fig4.add_trace(go.Scattermapbox(
        name='Centers of Gravity',
        mode = 'markers',
        hovertemplate = cogs['city'],
        lon = cogs['final_lon'],
        lat = cogs['final_lat'],
        marker = {'size': 10,
                    'color': color_options['cog']}))

    fig4.update_layout(
        margin ={'l':0,'t':0,'b':0,'r':0},
        mapbox = {
            'accesstoken': mapbox_access_token,
            'center': {'lon': df['final_lon'].mean(),
                        'lat': df['final_lat'].mean()},
            #'style': 'stamen-terrain',
            'zoom': getBoundsZoomLevel([df['final_lat'].max(),
                                        df['final_lon'].max(),
                                        df['final_lat'].min(),
                                        df['final_lon'].min()],
                                        {'height': map_height,
                                        'width': map_width})})

    fig4.update_layout(legend= {'itemsizing': 'constant'})


    # Create COG Table 
    pass

    graph_container_style = {'height': '26rem','width':'90vh','padding-bottom':'2rem'}

    return render_table(df,'ip_dt'), dbc.Container(dcc.Graph(figure=fig1,style={'height':'100%'}),style=graph_container_style), dash.no_update, dbc.Container(dcc.Graph(figure=fig4,style={'height':'100%'}),style=graph_container_style), render_table(cogs,'cog_table'),  settings_hash, True, "Good"
    # return render_table(df,'ip_dt'), dcc.Graph(figure=fig1) , dash.no_update, dcc.Graph(figure=fig4), render_table(cogs,'cog_table'),  settings_hash, True, "Good"

###########################
## Client Side Callbacks ##
###########################

# Output Panel radio button
app.clientside_callback(
    """
    function(output_navigator_value) {
    
        if(output_navigator_value==1){
            return [{"display":"flex"},{"display":"none"},{"display":"none"},{"display":"none"},{"display":"none"}];
        }else if(output_navigator_value==2){
            return [{"display":"none"},{"display":"flex"},{"display":"none"},{"display":"none"},{"display":"none"}];
        }else if(output_navigator_value==3){
            return [{"display":"none"},{"display":"none"},{"display":"flex"},{"display":"none"},{"display":"none"}];
        }else if(output_navigator_value==4){
            return [{"display":"none"},{"display":"none"},{"display":"none"},{"display":"flex"},{"display":"none"}];
        }else if(output_navigator_value==5){
            return [{"display":"none"},{"display":"none"},{"display":"none"},{"display":"none"},{"display":"flex"}];
        }
    }
    """,
    Output(component_id="input_data_table_panel",component_property="style"),
    Output(component_id="input_data_heatmap_panel",component_property="style"),
    Output(component_id="input_data_summary_panel",component_property="style"),
    Output(component_id="cog_map_panel",component_property="style"),
    Output(component_id="cog_table_panel",component_property="style"),
    Input(component_id='output_navigator', component_property='value'),
)

# Display the file name when a new file is uploaded
app.clientside_callback(
    """
    function(contents, filename) {
        if (typeof filename !== "undefined"){
            return [ filename , {'color':'green','font-weight':'500'}];
        }
        return [ 'Click Here To Upload Files' , {}];
    }
    """,
    Output(component_id='upload_file_text', component_property='children'),
    Output(component_id='upload_file_text', component_property='style'),
    Input(component_id='cog_input_excel', component_property='contents'),
    State(component_id='cog_input_excel', component_property='filename'),
    prevent_initial_call=True

)


# Check hahes and change border color
app.clientside_callback(
    """
    function(settings_hash, output_hash) {

        if (typeof output_hash !== "undefined"){
            if (settings_hash == output_hash){
                return {'border':'solid green 0.15rem','border-radius':'0.5rem',};
            }
            return {'border':'dashed red 0.15rem','border-radius':'0.5rem',};
        }
        return {'border':'solid black 0.15rem','border-radius':'0.5rem'};

    }
    """,
    Output(component_id='output_section', component_property='style'),
    Input(component_id='settings_hash', component_property='data'),
    Input(component_id='output_hash', component_property='data'),
    prevent_initial_call=True

)


# Open the help tab
app.clientside_callback(
    """
    function(n_clicks) {

        if (n_clicks % 2 == 0) {
            _style =  { "left": "50%","top": "50%","transform": "translate(-50%, -50%)","position":"fixed","display":"flex","justify-content":"center","align-items":"center","z-index":"1000", "min-height": "410px","box-shadow": "0 16px 24px 2px rgb(0 0 0 / 24%), 0 6px 30px 5px rgb(0 0 0 / 22%), 0 8px 10px -5px rgb(0 0 0 / 40%)","padding": "10px 0","width": "100%", "border-radius": "6px", "color": "rgba(0,0,0, 0.87)", "background": "#fff","height":"95%"}
            return _style
        }
        return {'display':'none'}

    }
    """,
    Output(component_id='help_tab', component_property='style'),
    Input(component_id='help_button', component_property='n_clicks'),
    prevent_initial_call=True

)


# Remove Map Image once a calcuation is done
app.clientside_callback(
    """
    function(output_hash) {
        if (typeof output_hash !== "undefined"){
            return {'display':'none'}
        }
    }
    """,
    Output(component_id='starter_image_panel', component_property='style'),
    Input(component_id='output_hash', component_property='data'),
    prevent_initial_call=True

)

if __name__ == "__main__":
    app.run_server(port=8005)
