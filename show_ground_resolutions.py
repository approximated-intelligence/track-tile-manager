"""
Web Mercator Ground Resolution Calculator

Calculates ground resolution (meters per pixel) across zoom levels and latitudes
for Web Mercator projection (EPSG:3857), commonly used in web mapping systems.

Ground resolution varies by:
- Zoom level: doubles with each increment
- Latitude: decreases toward poles due to Mercator distortion

Formula: resolution = (earth_circumference * cos(lat)) / (tile_size * 2^zoom)

Constants:
    earth_circumference: Earth's equatorial circumference in meters
    tile_size: Standard web map tile dimension (256x256 pixels)
"""

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
