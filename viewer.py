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
vol_dict = OrderedDict()
vol_dict['T1'] = load_volume_for_display('data/C3L-00016/t1.nii.gz')
vol_dict['T1ce'] = load_volume_for_display('data/C3L-00016/t1ce.nii.gz')
# vol_dict['T2'] = load_volume_for_display('data/C3L-00016/t2.nii.gz')
# vol_dict['FLAIR'] = load_volume_for_display('data/C3L-00016/flair.nii.gz')

ref_vol_name = list(vol_dict.keys())[0]
ref_vol = vol_dict[ref_vol_name]
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
axial_view = VolumeSlicer(app, np.asarray(ref_vol), spacing=ref_vol.voxel_size, axis=0)
coronal_view = VolumeSlicer(app, np.asarray(ref_vol), spacing=ref_vol.voxel_size, axis=1)
sagittal_view = VolumeSlicer(app, np.asarray(ref_vol), spacing=ref_vol.voxel_size, axis=2)
slice_views = html.Div(
    style={
        'display': 'grid',
        'gridTemplateColumns': '33% 33% 33%'
    },
    children=[
        html.Div([axial_view.graph, html.Br(), axial_view.slider, *axial_view.stores], style={'margin': '0 10px'}),
        html.Div([sagittal_view.graph, html.Br(), sagittal_view.slider, *sagittal_view.stores], style={'margin': '0 10px'}),
        html.Div([coronal_view.graph, html.Br(), coronal_view.slider, *coronal_view.stores], style={'margin': '0 10px'})
    ]
)

# volume selection
vol_checkboxes = dcc.RadioItems(
    id='vol-checkboxes',
    options=[{'label': vol_name, 'value': vol_name} for vol_name in vol_dict.keys()],
    value=ref_vol_name,
    style={'display': 'inline'}
)
vol_selection_div = html.Div(['Volume: ', vol_checkboxes], style={'display': 'inline'})

# main layout
app.layout = html.Div(
    children=[
        html.H1(case_name),
        slice_views,
        html.Br(),
        vol_selection_div,
        html.Br(),
        html.Br(),
        seg_selection_div
    ]
)


# callbacks
@app.callback(
    Output(axial_view.overlay_data.id, 'data'),
    Output(sagittal_view.overlay_data.id, 'data'),
    Output(coronal_view.overlay_data.id, 'data'),
    Input('seg-checkboxes', 'value')
)
def change_overlay(selected_seg_name):
    seg = np.asarray(seg_dict[selected_seg_name])
    return axial_view.create_overlay_data(seg, COLORMAP), sagittal_view.create_overlay_data(seg, COLORMAP), coronal_view.create_overlay_data(seg, COLORMAP)


# @app.callback(
#     Output(axial_view.graph.id, 'data'),
#     Output(sagittal_view.graph.id, 'data'),
#     Output(coronal_view.graph.id, 'data'),
#     Input('vol-checkboxes', 'value')
# )
# def change_volume(selected_vol_name):
#     vol = np.asarray(vol_dict[selected_vol_name])
#     return vol, vol, vol


if __name__ == '__main__':
    app.run_server(port=8000, host='127.0.0.1', debug=True, dev_tools_props_check=False)
