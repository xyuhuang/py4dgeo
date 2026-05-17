
import os
import re
from datetime import datetime
import trimesh
import numpy as np
import py4dgeo
from py4dgeo.util import Py4DGeoError

from datetime import datetime, timedelta
import trimesh
import numpy as np
import py4dgeo
from py4dgeo.util import Py4DGeoError

def read_obj_epochs_from_folder(folder: str, start_time: datetime, time_increment: timedelta):
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
            mesh = trimesh.load(path, process=False)
            epoch = py4dgeo.Epoch(cloud=mesh.vertices.copy())
            epoch.timestamp = current_time
            epochs.append(epoch)
            
            # Increment time for the next file
            current_time += time_increment
        except Exception as ex:
            print(f"Failed to read or process {fn}: {ex}")
            continue
            
    return epochs

