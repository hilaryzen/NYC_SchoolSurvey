import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans

df_survey = st.session_state['df_2014-15']
df_demloc = st.session_state['df_demloc']
df_qs = st.session_state['df_qs']

def clustering():
    # Normalize data
    scaler = MinMaxScaler()
    df_normal = pd.DataFrame(scaler.fit_transform(df_survey.drop(columns=['DBN', 'School Name', 'Year'])))
    # Complete missing values with kNN
    imputer = KNNImputer()
    df_trans = imputer.fit_transform(df_normal)

    # Find optimal # of clusters with elbow method
    inertia = []
    for k in range(2, 10):
        model = KMeans(n_clusters=k)
        model.fit(df_trans)
        inertia.append(model.inertia_)
    st.session_state['inertia'] = inertia

    # k-means clustering model with optimal # of clusters
    model = KMeans(n_clusters=3)
    model.fit(df_trans)
    df_survey['Cluster'] = model.predict(df_trans)
    df = pd.merge(df_survey, df_demloc, how='inner', on=['DBN', 'Year'])
    st.session_state['df_clustered'] = df

if 'df_clustered' not in st.session_state:
    clustering()

st.markdown("# Unsupervised Clustering")
# year = st.sidebar.selectbox(
#     'Choose survey year',
#      ('2014-15', '2015-16'))
year = '2014-15'
type = st.sidebar.selectbox(
    'Choose survey category or demographic factor',
     ('Student', 'Parent', 'Teacher', 'Student demographic'))

dem_dict = {'% Female Students': 'Female', '% Male Students': 'Male', '% Asian Students': 'Asian', '% Black Students': 'Black', '% Hispanic Students': 'Hispanic', '% Students who are Multiple Race Categories Not Represented': 'Multiple', '% White Students': 'White', '% Students with Disabilities': 'Disability', '% Students who are English Language Learners': 'ELL', '% Students in Poverty': 'Poverty', 'Economic Need Index': 'Need_index'}
if type == 'Student demographic':
    measure = st.sidebar.selectbox(
        'Choose demographic factor',
         ('% Female Students', '% Male Students', '% Asian Students', '% Black Students', '% Hispanic Students', '% Students who are Multiple Race Categories Not Represented', '% White Students', '% Students with Disabilities', '% Students who are English Language Learners', '% Students in Poverty', 'Economic Need Index'))
    code = dem_dict[measure]
else:
    measure = st.sidebar.selectbox(
        'Choose survey question',
         df_qs.loc[(df_qs['Year'] == year) & (df_qs['Group'] == type)]['QuesText'])
    code = df_qs.loc[(df_qs['Year'] == year) & (df_qs['Group'] == type) & (df_qs['QuesText'] == measure)]['QuesNum'].values[0]


st.markdown('Unsupervised clustering can be a useful method to group schools based on survey performance. The first algorithm used here is K-means clustering.')
st.markdown('The elbow method shows that the optimal number of clusters is 3. The graph below plots the inertia as the number of clusters goes from 2 to 10. The rate of decrease drops most sharply after k = 3.')
# Plot inertia for different # of clusters
fig = px.line(x = range(2,10), y = st.session_state['inertia'])
st.plotly_chart(fig, use_container_width=True)

st.markdown('---')
st.markdown('The graph below shows each school, separated into three different clusters by color. Choose a survey question or student demographic measure from the left sidebar to see how it compares between clusters.')
st.markdown('##### ' + measure)

map_df = st.session_state['df_clustered'][['LONGITUDE', 'LATITUDE', code, 'Cluster']].dropna(axis=0)
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
            get_fill_color=['Cluster == 0 ? 154 : (Cluster == 1 ? 68 : 15)', 'Cluster == 0 ? 3 : (Cluster == 1 ? 175 : 76)', 'Cluster == 0 ? 30 : (Cluster == 1 ? 105 : 92)', 140],
            radius=100,
            elevation_scale=50,
            pickable=True,
            auto_highlight=True
         )
     ],
     tooltip = {
        "html": "<b>%:</b> {" + code + "}",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white"
        }
    }
 ))
