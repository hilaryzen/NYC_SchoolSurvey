import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk

st.markdown("# Explore the NYC School Survey")

def load_data():
    # Read in 2014 student survey data
    df_s = pd.read_csv('data/2014-15_Student_Val.csv', encoding='UTF-8', encoding_errors='ignore').drop_duplicates()
    df_p = pd.read_csv('data/2014-15_Parent_Val.csv', encoding='UTF-8', encoding_errors='ignore').drop_duplicates()
    df_t = pd.read_csv('data/2014-15_Teacher_Val.csv', encoding='UTF-8', encoding_errors='ignore').drop_duplicates()
    df = pd.merge(df_s, df_p, how = 'outer', on = ['DBN'])
    df = pd.merge(df, df_t, how = 'outer', on = ['DBN'])
    df = df.drop(columns = ['School Name_x', 'School Name_y'])
    df['Year'] = '2014-15'
    st.session_state['df_2014-15'] = df
    # Check for null values in school name col
    print('Null school names: ', df['School Name'].isnull().any())
    # Read in question data dictionary
    st.session_state['df_qs'] = pd.read_csv('data/2014-19_Q.csv', encoding='latin-1')

    # Read in school demographics data
    df_demographics = pd.read_csv('data/2014-19_Demographics.csv')
    df_demographics['LOCATION_CODE'] = df_demographics['DBN'].apply(lambda x: x[2:])
    # Read in school location data
    df_loc = pd.read_csv('data/2019_School_Locations.csv')
    df_loc = df_loc[['LOCATION_CODE', 'LONGITUDE', 'LATITUDE']].drop_duplicates()
    # Merge demographics and location
    df_demloc = pd.merge(df_demographics, df_loc, how = 'inner', on = 'LOCATION_CODE')

    # Merge survey and demographic data
    st.session_state['df'] = pd.merge(df, df_demloc, how='inner', on=['DBN', 'Year'])

if 'df' not in st.session_state:
    load_data()

# df_dict = {'2014-15': {'Student': df_2014}}
df_qs = st.session_state['df_qs']
df = st.session_state['df']
#df_test = df[['School Name', 'LOCATION_CODE', 'LONGITUDE', 'LATITUDE', 'Economic_Need_Index']]
#df_test = df_test[df_test.isna().any(axis=1)]
#print(df_test)

# Add question select on sidebar
year = st.sidebar.selectbox(
    'Choose survey year',
     ('2014-15', '2015-16', '2016-17', '2017-18', '2018-19'))

group = st.sidebar.selectbox(
    'Choose survey respondents',
     ('Student', 'Parent', 'Teacher'))

ques = st.sidebar.selectbox(
    'Choose question',
     df_qs.loc[(df_qs['Year'] == year) & (df_qs['Group'] == group)]['QuesText'])

# Get question code for graph
code = df_qs.loc[(df_qs['Year'] == year) & (df_qs['Group'] == group) & (df_qs['QuesText'] == ques)]['QuesNum'].values[0]

# Map
map_df = df[['LONGITUDE', 'LATITUDE', 'Economic_Need_Index']].dropna(axis=0)
# print(map_df.head().to_string())
st.pydeck_chart(pdk.Deck(map_style=None,
     initial_view_state=pdk.ViewState(
         latitude=40.74,
         longitude=-73.98,
         zoom=10,
         pitch=50
     ),
     layers=[
         pdk.Layer(
            'ColumnLayer',
            data=map_df,
            get_position=['LATITUDE', 'LONGITUDE'],
            get_elevation='Economic_Need_Index',
            get_fill_color=['255 - Economic_Need_Index', '(1 - Economic_Need_Index) * 255', 0, 140],
            radius=100,
            elevation_scale=2000,
            #elevation_range=[0, 1000],
            pickable=True,
            auto_highlight=True
            #extruded=True,
         )
     ],
     tooltip = {
        "html": "<b>Elevation Value:</b> {Economic_Need_Index}",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white"
        }
    }
 ))

# Scatter plot using question selected on sidebar
fig = px.scatter(df[['Total Student Response Rate', code]], x='Total Student Response Rate', y=code, trendline='ols')
st.plotly_chart(fig, use_container_width=True)
