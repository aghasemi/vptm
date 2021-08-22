import pandas as pd
import numpy as np
import streamlit as st 
import plotly.express as px
import st_aggrid as stag
from traitlets.traitlets import default 
from plotly.validators.scatter.marker import SymbolValidator

df = pd.read_csv('pretrained-models.csv', index_col = 0)

st.set_page_config(page_title='Comparison of Vision Pretrained Models', layout='wide')

st.title( 'Comparison of Pre-trained Neural Network Architectures for Transfer Learning in Computer Vision Tasks')

st.markdown(''' 
Many neural network architectures have been proposed for solving computer vision tasks such as object recognition and image categorisation. Most of them are released along with pretrained weights obtained from training on ImageNet and similar datsets, making them a very good match for use as feature extractors in a transfer learning scenario for a much wider varierty of tasks.

With the very large number of neural network architectures and multiple pretrained modesl for each architectures (varying on training parameters, dataset, etc..), it is sometimes difficult to choose the right model to fine-tune or for transfer learning. The obvious factor to consider is the performance of the model on a benchmark dataset (usually ImageNet). However, not only the best-performing model on ImageNet is not necessarily the best as a feature extractor, but also there are other factors such as the size of the model (for storage purposes), and also the time complexity of inference with the model that come into play. One may simply prefer faster model to a more accurate one, especially when feature extraction would be executed on an embedded/edge device. Also specific to the case of transfer learning is the size of the feature vector computed by the neural network, which varies from less than 200 dimensions to more than 8000. In scenarios with a small training set, it is very desirable to have feature vectors as small as possible. 

For these reasons, and initially for personal use, I developed this buuble chart to more easily visualise and compare the differeces between various pretrained model available in the [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models) library. One can choose various dimensions for the horizntal and vertical axes as well as colour, size, and shape of the bubbles. It is also possible to filter out certain architectures.''')

#customize gridOptions
gb = stag.GridOptionsBuilder.from_dataframe(df)
gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)

gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)

symbols = list(set([sym.replace('-open','').replace('-dot','') for  sym in SymbolValidator().values if isinstance(sym, str) and not sym.isnumeric()]) -set(['x-thin' , 'cross-thin' , 'asterisk', 'hash',  'y-up', 'y-down','y-left','y-right', 'line-ew', 'line-nw','line-ns','line-ne']))
symbols_dot = list(set([sym for  sym in SymbolValidator().values if isinstance(sym, str) and not sym.isnumeric() and '-open' not in sym and '-dot' in sym]))
symbols += symbols_dot

families = sorted(list(set(df['Model Family'])))
families = st.multiselect(label='Please choose model families to show', options= families, default= families)


df = df[df['Model Family'].isin(families)]

cols = st.columns((1,1))

df['Input Size (Categorised)'] = df['Input Size'].apply(lambda x: str(x) if x in {224,256,384} else '<224' if x<224 else '>384' if x>384 else 'Other')
df['Log(Inference Time)'] = np.log(1+df["Inference Time"])

x_column = cols[0].selectbox('Please choose the paramater to show in the horizontal axis', options= ['Parameters', 'Size (MB)', 'ImageNet Top1 Error', 'Inference Time', 'Feature Vector Size', 'Input Size'], index = 0)
y_column = cols[0].selectbox('Please choose the paramater to show in the vertical axis', options= ['Parameters', 'Size (MB)', 'ImageNet Top1 Error', 'Inference Time', 'Feature Vector Size', 'Input Size'], index = 1)


color_column = cols[1].selectbox('Please choose the paramater to show as the colour of the blobs', options= ['Parameters', 'Size (MB)', 'ImageNet Top1 Error', 'Inference Time', 'Feature Vector Size', 'Input Size', 'Log(Inference Time)', 'None'], index = 3) #None is not a column name

symbol_column = cols[1].selectbox('Please choose the paramater to show as the shape of the blobs', options= ['Input Size (Categorised)', 'Model Family', 'None'], index = 1)

sub_cols = st.columns((2,1,1))

size_column = sub_cols[0].selectbox('Please choose the paramater to show as the size of the blobs', options= ['Parameters', 'Size (MB)', 'ImageNet Top1 Error', 'Inference Time', 'Feature Vector Size', 'Input Size', 'Log(Inference Time)', 'None'], index = 5)
x_is_log_scale = sub_cols[1].checkbox('X axis in log scale', False)
y_is_log_scale = sub_cols[2].checkbox('Y axis in log scale', False)

fig = px.scatter(df, x=x_column, y=y_column, 
                 size = None if size_column=='None' else size_column, symbol= None if symbol_column=='None' else symbol_column,
                 color= None if color_column=='None' else color_column,
                 log_y= y_is_log_scale, log_x= x_is_log_scale, height = 800, 
                 symbol_sequence= symbols, 
                 hover_name="Name", hover_data={'Model Family': True, 'Parameters': True, 'ImageNet Top1 Error': True, 'Inference Time': True, 'Feature Vector Size': True, 'Input Size': True, 'Log(Inference Time)': False, 'Input Size (Categorised)': False})
#fig.layout.legend.x = 1.1
#fig.layout.coloraxis.colorbar.y = .55
fig.update_layout({'legend_orientation':'h'})

st.plotly_chart(fig, use_container_width=True)

st.markdown('Visualised Data')
stag.AgGrid(
    df[['Name','Model Family', 'Parameters', 'ImageNet Top1 Error', 'Inference Time', 'Feature Vector Size', 'Input Size']], 
    height='800px', 
    width='100%',
    fit_columns_on_grid_load=True)