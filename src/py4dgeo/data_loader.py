
import os
from datetime import datetime, timedelta
import trimesh
import numpy as np
import py4dgeo


def read_obj_and_assign_timestamps(folder: str, start_time: datetime, time_increment: timedelta):
    """
    Reads all .obj files from a folder, sorts them, and assigns timestamps sequentially.
    """
    # Find all .obj files and sort them alphabetically to ensure consistent order
    try:
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.obj')])
    except FileNotFoundError:
        print(f"Error: Directory not found at {folder}")
        return []
        
    if not files:
        print(f"Warning: No .obj files found in {folder}")
        return []
        
    epochs = []
    current_time = start_time
    for fn in files:
        path = os.path.join(folder, fn)
        try:
            mesh = trimesh.load(path, force='mesh', process=False)
            epoch = py4dgeo.Epoch(cloud=mesh.vertices.copy())
            epoch.timestamp = current_time
            epochs.append(epoch)
            
            # Increment time for the next file
            current_time += time_increment
        except Exception as ex:
            print(f"Failed to read or process {fn}: {ex}")
            continue
            
    return epochs


def read_pc_epochs_and_assign_timestamps(folder: str, start_time: datetime, time_increment: timedelta):
    """
    Reads all .las, .laz, and .xyz files from a folder, sorts them, 
    and assigns timestamps sequentially using py4dgeo's native readers.
    """
    valid_extensions = ('.las', '.laz', '.xyz')
    
    try:
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith(valid_extensions)])
    except FileNotFoundError:
        print(f"Error: Directory not found at {folder}")
        return []
        
    if not files:
        print(f"Warning: No point cloud files (.las, .laz, .xyz) found in {folder}")
        return []
        
    epochs = []
    current_time = start_time
    
    for fn in files:
        path = os.path.join(folder, fn)
        ext = os.path.splitext(fn)[1].lower() 
        
        try:
            if ext in ['.las', '.laz']:
                epoch = py4dgeo.read_from_las(path)
            elif ext == '.xyz':
                epoch = py4dgeo.read_from_xyz(path)
            else:
                continue 
                
            epoch.timestamp = current_time
            epochs.append(epoch)

            current_time += time_increment
            
        except Exception as ex:
            print(f"Failed to read or process {fn}: {ex}")
            continue
            
    return epochs