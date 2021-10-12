from os import path
from collections import OrderedDict
import numpy as np
import dash
import plotly.express as px
from dash import html
from dash_slicer import VolumeSlicer
import dash_core_components as dcc
from dash.dependencies import Input, Output
from hlxpy import io
from hlxpy.volume import transpose, flip, vol_resample
import PIL


def load_volume_for_display(uri):
    vol = io.volread(uri)
    vol = flip(transpose(vol, (2,1,0)), 0)
    return vol


def load_seg_for_display(uri, ref_vol):
    seg = load_volume_for_display(uri)
    seg = vol_resample(seg, ref_vol, order=0).astype(np.uint8)
    seg[seg == 3] = 1
    return seg


# load data
vol = load_volume_for_display('data/C3L-00016/t1.nii.gz')
seg_dict = OrderedDict()
seg_dict['Consensus'] = load_seg_for_display('data/C3L-00016/tumor-seg-consensus.nii.gz', vol)
seg_dict['Seibert'] = load_seg_for_display('data/C3L-00016/tumor-seg-seibert.nii.gz', vol)
seg_dict['Rudie'] = load_seg_for_display('data/C3L-00016/tumor-seg-rudie.nii.gz', vol)
seg_dict['Ziseen'] = load_seg_for_display('data/C3L-00016/tumor-seg-ziseen.nii.gz', vol)


# creeate app + layout
app = dash.Dash(__name__, update_title=None)

axial_view = VolumeSlicer(app, np.asarray(vol), spacing=vol.voxel_size, axis=2)

seg_checkboxes = dcc.RadioItems(
    id='seg-checkboxes',
    options=[{'label': seg_name, 'value': seg_name} for seg_name in seg_dict.keys()],
    value='Consensus'
)

app.layout = html.Div(
    style={
        'display': 'grid',
        'gridTemplateColumns': '33% 33%',
    },
    children=[
        html.Div([
            seg_checkboxes,
            axial_view.graph, 
            html.Br(), 
            axial_view.slider, 
            *axial_view.stores]
        ),
    ],
)


@app.callback(
    Output(axial_view.overlay_data.id, 'data'),
    [Input('seg-checkboxes', 'value')],
)
def toggle_overlay(selected_seg_name):
    colormap = [(255, 255, 0, 50), (255, 0, 0, 100)]
    seg = np.asarray(seg_dict[selected_seg_name])
    return axial_view.create_overlay_data(seg, colormap)



if __name__ == '__main__':
    app.run_server(port=8000, host='127.0.0.1', debug=True, dev_tools_props_check=False)
