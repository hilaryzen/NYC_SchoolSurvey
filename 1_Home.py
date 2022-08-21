import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk

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
    percent_cols = ['% Female Students', '% Male Students', '% Asian Students', '% Black Students', '% Hispanic Students', '% Students who are Multiple Race Categories Not Represented', '% White Students', '% Students with Disabilities', '% Students who are English Language Learners', '% Students in Poverty', 'Economic Need Index']
    for col in percent_cols:
        df_demographics[col] = df_demographics[col].multiply(100).round(0)
    df_demographics.to_csv('dem_test.csv')
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

# Add question select on sidebar
# year = st.sidebar.selectbox(
#     'Choose survey year',
#      ('2014-15', '2015-16'))
year = '2014-15'

group = st.sidebar.selectbox(
    'Choose survey respondents',
     ('Student', 'Parent', 'Teacher'))

ques = st.sidebar.selectbox(
    'Choose survey question',
     df_qs.loc[(df_qs['Year'] == year) & (df_qs['Group'] == group)]['QuesText'])

dem = st.sidebar.selectbox(
    'Choose demographic factor',
     ('% Female Students', '% Male Students', '% Asian Students', '% Black Students', '% Hispanic Students', '% Students who are Multiple Race Categories Not Represented', '% White Students', '% Students with Disabilities', '% Students who are English Language Learners', '% Students in Poverty', 'Economic Need Index'))

# Get question code for graph
code = df_qs.loc[(df_qs['Year'] == year) & (df_qs['Group'] == group) & (df_qs['QuesText'] == ques)]['QuesNum'].values[0]

st.markdown('# Explore the NYC School Survey')
st.markdown('Created by Hilary Zen')
st.markdown('Every year, the NYC School Survey is given to all parents, all teachers, and students in grades 6 to 12. This Streamlit app uses datasets from [NYC OpenData](https://opendata.cityofnewyork.us/) to analyze survey results from the 2014-2015 school year, and uncover correlations between survey performance and school demographics.')
st.markdown('On the left sidebar, you can select the survey group (students, parents, or teachers), a survey question, and a demographic factor to compare against.')
st.markdown('---')
st.markdown('##### ' + ques)
st.markdown('Positive responses: ' + df_qs.loc[df_qs['QuesNum'] == code]['PosAns'].values[0])
st.markdown('Negative responses: ' + df_qs.loc[df_qs['QuesNum'] == code]['NegAns'].values[0])

st.markdown('The map below shows each school as a column. Taller columns show a higher percentage of positive responses to the survey question, while the color indicates the demographic percentage. Light yellow means that the demographic is close to 0% of the student body, while dark red corresponds to 100%.')

# Map
map_df = df[['LONGITUDE', 'LATITUDE', dem, code]].dropna(axis=0)
map_df['Color_r'] = df[dem] / 100
map_df['Color_g'] = (1 - df[dem] / 100) * 255
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
            get_position=['LONGITUDE', 'LATITUDE'],
            get_elevation=code,
            get_fill_color=['255 - Color_r', 'Color_g', 0, 140],
            radius=100,
            elevation_scale=50,
            pickable=True,
            auto_highlight=True
         )
     ],
     tooltip = {
        "html": "<b>Demographic %:</b> {" + dem + "}<br> <b>Positive Answer %:</b> {" + code + "}",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white"
        }
    }
 ))

st.markdown('---')
st.markdown('The scatterplot below shows the demographic factor on the x axis and the survey positive response % on the y axis.')

# Scatter plot using question selected on sidebar
fig = px.scatter(df[['School Name', dem, code]], x=dem, y=code, trendline='ols', hover_name='School Name')
st.plotly_chart(fig, use_container_width=True)
