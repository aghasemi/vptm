import pandas as pd
import numpy as np
import streamlit as st 
import plotly.express as px
import st_aggrid as stag
from traitlets.traitlets import default 

df = pd.read_csv('pretrained-models.csv', index_col = 0)

st.set_page_config(page_icon='Comparison of Vision Pretrained Models', layout='wide')

#customize gridOptions
gb = stag.GridOptionsBuilder.from_dataframe(df)
gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)

gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)

symbols = ['circle', 'square', 'diamond', 'cross', 'triangle-up', 'triangle-down', 'triangle-left', 'triangle-right', 
                                  'triangle-ne', 'triangle-nw', 'triangle-sw', 'triangle-se', 'pentagon', 'hexagon', 'hexagon2', 'hexagram', 'star', 'octagon']


families = sorted(list(set(df['Model Family'])))
families = st.multiselect(label='Please choose model families to show', options= families, default= families)


df = df[df['Model Family'].isin(families)]

cols = st.columns((1,1))

df['Input Size (Categorised)'] = df['Input Size'].apply(lambda x: str(x) if x in {224,256,384} else '<224' if x<224 else '>384' if x>384 else 'Other')
df['Log(Inference Time)'] = np.log(1+df["Inference Time"])

x_column = cols[0].selectbox('Please choose the paramater to show in the horizontal axis', options= ['Parameters', 'ImageNet Top1 Error', 'Inference Time', 'Feature Vector Size', 'Input Size'], index = 0)
y_column = cols[0].selectbox('Please choose the paramater to show in the vertical axis', options= ['Parameters', 'ImageNet Top1 Error', 'Inference Time', 'Feature Vector Size', 'Input Size'], index = 1)


color_column = cols[1].selectbox('Please choose the paramater to show as the colour of the blobs', options= ['Parameters', 'ImageNet Top1 Error', 'Inference Time', 'Feature Vector Size', 'Input Size', 'Log(Inference Time)'], index = 3)

symbol_column = cols[1].selectbox('Please choose the paramater to show as the shape of the blobs', options= ['Input Size (Categorised)', 'Model Family'], index = 1)

sub_cols = st.columns((2,1,1))

size_column = sub_cols[0].selectbox('Please choose the paramater to show as the size of the blobs', options= ['Parameters', 'ImageNet Top1 Error', 'Inference Time', 'Feature Vector Size', 'Input Size', 'Log(Inference Time)'], index = 5)
x_is_log_scale = sub_cols[1].checkbox('X axis in log scale', False)
y_is_log_scale = sub_cols[2].checkbox('Y axis in log scale', False)

fig = px.scatter(df, x=x_column, y=y_column, size=size_column, symbol= symbol_column, color= color_column ,log_y= y_is_log_scale, log_x= x_is_log_scale, height = 800, 
                 symbol_sequence= symbols, 
                 hover_name="Name", hover_data={'Model Family': True, 'Parameters': True, 'ImageNet Top1 Error': True, 'Inference Time': True, 'Feature Vector Size': True, 'Input Size': True, 'Log(Inference Time)': False, 'Input Size (Categorised)': False})
#fig.layout.legend.x = 1.1
#fig.layout.coloraxis.colorbar.y = .55
fig.update_layout({'legend_orientation':'h'})

st.plotly_chart(fig, use_container_width=True)

stag.AgGrid(
    df[['Name','Model Family', 'Parameters', 'ImageNet Top1 Error', 'Inference Time', 'Feature Vector Size', 'Input Size']], 
    height='800px', 
    width='100%',
    fit_columns_on_grid_load=True)