import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans

st.markdown("# Unsupervised Clustering")
st.sidebar.markdown("# School Performance")

df = st.session_state['df_2014-15']
# Normalize data
scaler = MinMaxScaler()
df_normal = pd.DataFrame(scaler.fit_transform(df.drop(columns=['DBN', 'School Name', 'Year'])))
# print(df_normal.head())

# Complete missing values with kNN
imputer = KNNImputer()
df_trans = imputer.fit_transform(df_normal)

# Find optimal # of clusters with elbow method
inertia = []
for k in range(2, 10):
    model = KMeans(n_clusters=k)
    model.fit(df_trans)
    inertia.append(model.inertia_)

# Plot inertia for different # of clusters
fig = px.line(x = range(2,10), y = inertia)
st.plotly_chart(fig, use_container_width=True)

# k-means clustering model with optimal # of clusters
model = KMeans(n_clusters=3)
model.fit(df_trans)
