import numpy as np
import pandas as pd

# Constants
earth_circumference = 40075016.686  # meters
tile_size = 256  # pixels
latitudes = [0, 30, 45, 60, 75]  # Different latitudes for the table
zoom_levels = range(4, 20)  # Zoom levels from 4 to 19

# Prepare the data
data = []

for lat in latitudes:
    cos_lat = np.cos(
        np.radians(lat)
    )  # Adjust resolution for the Earth's curvature at this latitude
    for zoom in zoom_levels:
        num_pixels = tile_size * (2**zoom)  # Total number of pixels at this zoom level
        resolution = (
            earth_circumference * cos_lat / num_pixels
        )  # Ground resolution at this zoom level and latitude
        data.append([zoom, lat, resolution])

# Create a DataFrame for easy viewing
df = pd.DataFrame(data, columns=["Zoom Level", "Latitude", "Ground Resolution"])

# Pivot the table to show latitudes as columns and zoom levels as rows
df_pivot = df.pivot(index="Zoom Level", columns="Latitude", values="Ground Resolution")

# Print the DataFrame to display it
print(df_pivot)
