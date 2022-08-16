import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# Read in 2014 student survey data
df_survey_2014 = pd.read_csv('data/2014-15_Student_Val.csv')
df_qs_2014 = pd.read_csv('data/2014-15_Student_Q.csv')

# Add question select on sidebar
option = st.sidebar.selectbox(
    'Choose question option',
     df_qs_2014['Option'])

fig = px.scatter(df_survey_2014[['Response_Rate', option]], x='Response_Rate', y=option, trendline='ols')
st.plotly_chart(fig, use_container_width=True)
