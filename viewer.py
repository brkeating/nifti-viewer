from os import path
from collections import OrderedDict
import numpy as np
import plotly.express as px
import dash
from dash import html
from dash_slicer import VolumeSlicer
import dash_core_components as dcc
from dash.dependencies import Input, Output
from hlxpy import io
from hlxpy.volume import transpose, flip, vol_resample


COLORMAP = px.colors.qualitative.D3  # https://plotly.com/python/discrete-color/
SEGMENTATION_LABEL = 'Brain Tumor Overlay'



def load_volume_for_display(uri):
    vol = io.volread(uri)
    vol = flip(transpose(vol, (2,1,0)), 0)
    return vol


def load_seg_for_display(uri, ref_vol):
    seg = load_volume_for_display(uri)
    seg = vol_resample(seg, ref_vol, order=0).astype(np.uint8)
    return seg



# load data
case_name = 'C3L-00016'
T1_vol = load_volume_for_display('data/C3L-00016/t1.nii.gz')
T1ce_vol = load_volume_for_display('data/C3L-00016/t1ce.nii.gz')
T2_vol = load_volume_for_display('data/C3L-00016/t2.nii.gz')
FLAIR_vol = load_volume_for_display('data/C3L-00016/flair.nii.gz')

ref_vol = T1_vol
seg_dict = OrderedDict()
seg_dict['None'] = np.zeros_like(ref_vol, dtype=np.uint8)
seg_dict['Consensus'] = load_seg_for_display('data/C3L-00016/tumor-seg-consensus.nii.gz', ref_vol)
# seg_dict['Seibert'] = load_seg_for_display('data/C3L-00016/tumor-seg-seibert.nii.gz', ref_vol)
# seg_dict['Rudie'] = load_seg_for_display('data/C3L-00016/tumor-seg-rudie.nii.gz', ref_vol)
# seg_dict['Ziseen'] = load_seg_for_display('data/C3L-00016/tumor-seg-ziseen.nii.gz', ref_vol)



# creeate app + layout
app = dash.Dash(__name__, update_title=None)
app.title = 'Cortechs Image Viewer'

# widgets for selecting segmentation overlay
seg_checkboxes = dcc.RadioItems(
    id='seg-checkboxes',
    options=[{'label': seg_name, 'value': seg_name} for seg_name in seg_dict.keys()],
    value='Consensus',
    style={'display': 'inline'}
)
seg_selection_div = html.Div([SEGMENTATION_LABEL+': ', seg_checkboxes], style={'display': 'inline'})



# slice views
T1_view = VolumeSlicer(app, np.asarray(T1_vol), spacing=T1_vol.voxel_size, axis=0)
T1ce_view = VolumeSlicer(app, np.asarray(T1ce_vol), spacing=T1ce_vol.voxel_size, axis=0)
T2_view = VolumeSlicer(app, np.asarray(T2_vol), spacing=T2_vol.voxel_size, axis=0)
FLAIR_view = VolumeSlicer(app, np.asarray(FLAIR_vol), spacing=FLAIR_vol.voxel_size, axis=0)
slice_views = html.Div(
    style={
        'display': 'grid',
        'gridTemplateColumns': '50% 50%'
    },
    children=[
        html.Div([
            html.Div([html.H3('T1', style={'text-align': 'center'}), T1_view.graph, html.Br(), T1_view.slider, *T1_view.stores], style={'margin': '20px'}),
            html.Div([html.H3('T1ce', style={'text-align': 'center'}), T1ce_view.graph, html.Br(), T1ce_view.slider, *T1ce_view.stores], style={'margin': '20px'})
        ]),
        html.Div([
            html.Div([html.H3('T2', style={'text-align': 'center'}), T2_view.graph, html.Br(), T2_view.slider, *T2_view.stores], style={'margin': '20px'}),
            html.Div([html.H3('FLAIR', style={'text-align': 'center'}), FLAIR_view.graph, html.Br(), FLAIR_view.slider, *FLAIR_view.stores], style={'margin': '20px'})
        ])
    ]
)

# main layout
app.layout = html.Div(
    children=[
        html.H1(case_name),
        seg_selection_div,
        html.Br(),
        slice_views
    ]
)


# callbacks
@app.callback(
    Output(T1_view.overlay_data.id, 'data'),
    Output(T1ce_view.overlay_data.id, 'data'),
    Output(T2_view.overlay_data.id, 'data'),
    Output(FLAIR_view.overlay_data.id, 'data'),
    Input('seg-checkboxes', 'value')
)
def change_overlay(selected_seg_name):
    seg = np.asarray(seg_dict[selected_seg_name])
    return T1_view.create_overlay_data(seg, COLORMAP), T1ce_view.create_overlay_data(seg, COLORMAP), T2_view.create_overlay_data(seg, COLORMAP), FLAIR_view.create_overlay_data(seg, COLORMAP)



if __name__ == '__main__':
    app.run_server(port=8000, host='127.0.0.1', debug=True, dev_tools_props_check=False)
