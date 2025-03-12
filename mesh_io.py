from glob import glob
from os.path import join as pjoin, split as psplit, abspath, basename, dirname, isfile, exists

# loading
import nibabel as nib
import numpy as np

# pyvista
import pyvista as pv

def read_raw(file, dtype='double'):
    '''load .raw file'''
    
    return np.fromfile(file, dtype=dtype)

def load_freesurfer_annot(annot_file):
    '''load freesurfer .annot file'''
    label, ctab, names = nib.freesurfer.io.read_annot(annot_file)
    return label, ctab, names


def load_freesurfer_n(freesurfer_file):
    '''load mesh file'''
    vertices, faces, create_stamp, volume_info = nib.freesurfer.io.read_geometry(freesurfer_file, read_metadata=True, read_stamp=True)
    return vertices, faces, create_stamp, volume_info

def extract_surface_roi_feature(annot_file, raw_file, hemi):
    '''
    Extract the ROI average feature on cortical surface (raw_file)
    '''
    if '.raw' in raw_file:
        feature = read_raw(raw_file, dtype='float32')
    else:
        feature = load_mgh_scalers(raw_file)
        
    annot_labels, _, annot_names = load_freesurfer_annot(annot_file)
    
    surface_roi_feature_dict = {}
    for label in list(set(annot_labels)):
        if label != -1:
            # print(label, annot_names[label], f'ctx-{hemi}-{annot_names[label].decode("utf-8")}', np.mean(feature[annot_labels==label]))
            surface_roi_feature_dict[f'ctx-{hemi}-{annot_names[label].decode("utf-8")}'] = np.mean(feature[annot_labels==label])
            
    return surface_roi_feature_dict


def load_freesurfer2pyvista(surf_file, 
                            scalar_file=None, scalar_min=0, scalar_max=10,
                            annot_file=None):
    
    '''load freesurfer to pyvista polydata'''
    
    
    vertices, faces, _, _ = load_freesurfer_n(surf_file)
    
    # using pv.PolyData for mesh
    # mesh = pv.PolyData.from_irregular_faces(vertices, faces) 
    mesh = pv.PolyData.from_regular_faces(vertices, faces)
    
    # load scalar
    if scalar_file is not None:
        scalar_data = read_raw(scalar_file, dtype='float32')
        scalar_data = np.clip(scalar_data, scalar_min, scalar_max) # clip the pet 
        
        # mesh.scalar = scalar_data
        mesh['scalars'] = scalar_data
    
    # load annotation
    if annot_file is not None:
        annot_labels, _, annot_names = load_freesurfer_annot(annot_file)
        
        mesh.annot_labels = annot_labels
        mesh.annot_names = annot_names
        
    return mesh


def laplacian_smoothing(vertices, faces, scalar, iterations=10, lambda_factor=0.5):
    """
    Perform Laplacian smoothing on a function defined over a mesh.

    Parameters:
    - vertices, faces: mesh: vertices and faces
    - function_values: np.array, the function values defined at each vertex.
    - iterations: int, number of smoothing iterations.
    - lambda_factor: float, smoothing factor (0 < lambda_factor < 1).

    Returns:
    - smoothed_values: np.array, the smoothed function values.
    """

    n_vertices = len(vertices)
    # Create a list of neighboring vertices for each vertex
    neighbors = [set() for _ in range(n_vertices)]
    for face in faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    neighbors[face[i]].add(face[j])
    
    # Convert sets to lists for easier indexing
    neighbors = [list(neighbor_set) for neighbor_set in neighbors]
    
    smoothed_values = np.copy(scalar)
    
    for _ in range(iterations):
        new_values = np.copy(smoothed_values)
        for i in range(n_vertices):
            if len(neighbors[i]) > 0:
                # Compute the average of the neighboring values
                neighbor_avg = np.mean(smoothed_values[neighbors[i]])
                # Update the value using the Laplacian smoothing formula
                new_values[i] = smoothed_values[i] + lambda_factor * (neighbor_avg - smoothed_values[i])
        smoothed_values = new_values
    
    return smoothed_values
    