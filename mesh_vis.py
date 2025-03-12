# loading
import nibabel as nib
import numpy as np

# display
import scipy.io as spio
import plotly.graph_objects as go
from scipy.spatial.distance import cosine, cdist, directed_hausdorff

from plotly.subplots import make_subplots
from plotly.io import write_image

# def plot_patch(v, f, scalars, color='red', colorscale='Rainbow'):    
#     '''
#     Display scalars on mesh using plotly
#     '''
#     fig = go.Figure(data=[go.Mesh3d(x=v[:,0], y=v[:,1], z=v[:,2],
#                                     i=f[:,0], j=f[:,1], k=f[:,2],
#                                     color=color, opacity=1, intensity = scalars, colorscale=colorscale)])
    
#     fig.show()
    


def plot_patch(v, f, scalars, color='red', colorscale='Rainbow', view='right_lateral', figname=None):
    """
    Display scalars on mesh using plotly, with configurable view.

    :param v: Vertices of the mesh
    :param f: Faces of the mesh
    :param scalars: Scalar values for each vertex
    :param color: Color of the mesh
    :param colorscale: Color scale for scalar values
    :param view: String indicating the desired view ('right_lateral', 'left_lateral', etc.)
    """
    # Camera settings based on the view
    camera_views = {
        'right_lateral': dict(eye=dict(x=2.5, y=0, z=0)),
        'left_lateral': dict(eye=dict(x=-2.5, y=0, z=0)),
        'top_down': dict(eye=dict(x=0, y=0, z=2.5)),
        'bottom_up': dict(eye=dict(x=0, y=0, z=-2.5)),
        'frontal': dict(eye=dict(x=0, y=2.5, z=0)),
        'posterior': dict(eye=dict(x=0, y=-2.5, z=0))
    }

    # Create the figure with the specified mesh and scalar data
    fig = go.Figure(data=[go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        color=color, opacity=1, intensity=scalars, colorscale=colorscale
    )])

    # Update the layout with the selected camera view
    if view in camera_views:
        # fig.update_layout(scene_camera=camera_views[view])
            # Update the layout to remove grids and apply the selected camera view
        fig.update_layout(
            scene=dict(
                xaxis_showgrid=False, yaxis_showgrid=False, zaxis_showgrid=False,
                xaxis=dict(showbackground=False, gridcolor='rgba(0,0,0,0)'),
                yaxis=dict(showbackground=False, gridcolor='rgba(0,0,0,0)'),
                zaxis=dict(showbackground=False, gridcolor='rgba(0,0,0,0)'),
                xaxis_title='',  # Removes x-axis title
                yaxis_title='',  # Removes y-axis title
                zaxis_title='',  # Removes z-axis title
                xaxis_showticklabels=False,  # Hides x-axis tick labels
                yaxis_showticklabels=False,  # Hides y-axis tick labels
                zaxis_showticklabels=False,   # Hides z-axis tick labels
                camera=camera_views.get(view, camera_views[view]))
            # title='',  # Removes any title at the top of the plot
            # # showlegend=False  # Optional: remove legend if present
        )
    else:
        raise ValueError("Invalid view specified. Choose from 'right_lateral', 'left_lateral', etc.")
    
    if figname is not None:
        write_image(fig, figname.replace('.png', f'_{view}.png'))
    else:
        fig.show()

# Example of calling the function with a view
# plot_patch(vertices, faces, intensity_values, view='right_lateral')




def plot_patch_multiview(v, f, scalars, color='red', colorscale='Rainbow', cmin=0, cmax=2, views=None, figtitle='', figname=None):
    """
    Display multiple views of scalars on mesh using Plotly in one figure.

    :param v: Vertices of the mesh
    :param f: Faces of the mesh
    :param scalars: Scalar values for each vertex
    :param color: Color of the mesh
    :param colorscale: Color scale for scalar values
    :param views: List of views as strings (e.g., ['right_lateral', 'left_lateral', 'top_down'])
    """
    if views is None:
        views = ['right_lateral', 'left_lateral', 'top_down', 'bottom_up', 'frontal', 'posterior']

    # Camera settings based on common views
    camera_views = {
        'right_lateral': dict(eye=dict(x=2.5, y=0, z=0)),
        'left_lateral': dict(eye=dict(x=-2.5, y=0, z=0)),
        'top_down': dict(eye=dict(x=0, y=0, z=2.5)),
        'bottom_up': dict(eye=dict(x=0, y=0, z=-2.5)),
        'frontal': dict(eye=dict(x=0, y=2.5, z=0)),
        'posterior': dict(eye=dict(x=0, y=-2.5, z=0))
    }

    # Setup subplot grid
    rows = 2  # Adjust based on the number of views
    cols = 3  # Adjust based on the number of views
    specs = [[{'type': 'scene'}]*cols]*rows
    fig = make_subplots(
        rows=rows, cols=cols,
        specs=specs,
        subplot_titles=views
    )

    # Add a plot for each view
    for i, view in enumerate(views):
        row = i // cols + 1
        col = i % cols + 1
        camera_setting = camera_views.get(view, camera_views['right_lateral'])

        fig.add_trace(
            go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                color=color, opacity=1, intensity=scalars, colorscale=colorscale, cmin=cmin, cmax=cmax,  # Set the maximum value for the color scale
                showscale=(i == 0)  # Only show scale on the first subplot to avoid repetition
            ),
            row=row, col=col
        )
        fig.update_scenes(
            camera=dict(eye=camera_setting['eye']),
            xaxis_showgrid=False, yaxis_showgrid=False, zaxis_showgrid=False,
            xaxis_showbackground=False, yaxis_showbackground=False, zaxis_showbackground=False,
            xaxis_title='',  # Removes x-axis title
            yaxis_title='',  # Removes y-axis title
            zaxis_title='',  # Removes z-axis title
            xaxis_showticklabels=False,  # Hides x-axis tick labels
            yaxis_showticklabels=False,  # Hides y-axis tick labels
            zaxis_showticklabels=False,   # Hides z-axis tick labels
            row=row, col=col
        )

    # Update layout to better display the subplots
    fig.update_layout(
        height=900, width=1200,
        title_text=figtitle
    )
    
    if figname is not None:
        write_image(fig, figname)
    else:
        fig.show()

# Example of calling the function with specific views
# plot_patch(vertices, faces, intensity_values, views=['right_lateral', 'left_lateral', 'top_down'])
