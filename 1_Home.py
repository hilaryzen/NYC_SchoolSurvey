import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.markdown("# Explore the NYC School Survey")

def load_data():
    # Read in 2014 student survey data
    df_2014 = pd.read_csv('data/2014-15_Student_Val.csv', encoding='latin-1')
    df_2014['Year'] = '2014-15'
    st.session_state['df_2014'] = df_2014
    # Read in question data dictionary
    st.session_state['df_qs'] = pd.read_csv('data/2014-15_Student_Q.csv')

    # Read in school demographics data
    df_demographics = pd.read_csv('data/2014-19_Demographics.csv')
    # Merge survey and demographic data
    st.session_state['df'] = pd.merge(df_2014, df_demographics, how='inner', on=['DBN', 'Year'])

if 'df' not in st.session_state:
    load_data()

# df_dict = {'2014-15': {'Student': df_2014}}
df_qs = st.session_state['df_qs']
df = st.session_state['df']

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

# Scatter plot using question selected on sidebar
fig = px.scatter(df[['Total Student Response Rate', code]], x='Total Student Response Rate', y=code, trendline='ols')
st.plotly_chart(fig, use_container_width=True)
