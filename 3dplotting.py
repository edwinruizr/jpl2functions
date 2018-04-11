
# coding: utf-8

# In[2]:


import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

import pandas as pd

def make3dPlot(dataframe):
    #plotly.offline.init_notebook_mode()
    # Read data from a csv
    z_data = dataframe
    colnames = (dataframe.columns.values)
    data = [
       go.Surface(
           z=z_data.as_matrix()
       )
    ]
    layout = go.Layout(
       title=colnames[2],
       autosize=False,
       width=500,
       height=500,
       margin=dict(
           l=65,
           r=50,
           b=65,
           t=90
       )
    )
    fig = go.Figure(data=data, layout=layout)
    plot(fig)
    return

column_names = ['latitude','longitude','elevation']
z_data = pd.read_csv('./temperature_xyz/resizedLOLA.xyz', 
                       delim_whitespace=True, names=column_names, usecols=['latitude', 'longitude', 'elevation'])

make3dPlot(z_data)

