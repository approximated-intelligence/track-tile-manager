# Track Tile Downloader

Download map tiles along a GPX track for offline use in OsmAnd and other mapping apps.

## What it does

Takes a GPX track and downloads XYZ map tiles in a configurable buffer zone around the route. Outputs MBTiles format for use in various mapping applications.

Useful for preparing offline maps for hiking/biking trips, caching tiles for areas with poor connectivity, or creating custom map extracts for specific routes.

## Features

- Geodetically accurate buffering using oblique Mercator projection
- Configurable buffer distance and zoom levels
- Two additional tile expansion strategies for better coverage
- Optional JPEG compression to save space
- Automatic retry with exponential backoff on errors
- Persistent tile cache to avoid re-downloading

## Installation

Requires Python 3.8+. Use with `uv`:

```bash
# Run directly without installing
uvx --from https://github.com/project-url download-track-tiles https://tile.openstreetmap.org track.gpx output.mbtiles

# Or install dependencies and run
uv sync
uv run track_tile_download.py https://tile.openstreetmap.org track.gpx output.mbtiles
```

## Usage

Basic usage:

```bash
uv run download_track_tiles.py https://tile.openstreetmap.org track.gpx output.mbtiles
```

This creates `output.mbtiles` with tiles covering your track plus a 500m buffer zone.

The default URL format is `{url}/{z}/{x}/{y}.png`, which constructs tile URLs like `https://tile.openstreetmap.org/12/2048/1024.png`.

### Options

```
-b, --buffer METERS       Buffer distance from track (default: 500m)
-z MIN MAX                Zoom level range (default: 4 16)
--corner-expand           Add corner-based tile expansion (adaptive)
--tile-buffer N           Add N-tile uniform buffer around all tiles
--jpg QUALITY             Convert tiles to JPEG (1-100, saves space)
-f, --force               Overwrite an existing output file
--url-format FORMAT       Custom URL format string (default: {url}/{z}/{x}/{y}.png)

base_url                  Base URL for tile server (e.g., https://tile.server.com)
gpx_path                  Input GPX file
output_filename           Output filename (with extension)
```

### URL formatting

The downloader uses two parameters for tile URLs:

- `base_url`: The base URL passed as a positional argument (e.g., `https://tile.openstreetmap.org`)
- `--url-format`: A format string that combines the base URL with tile coordinates

The format string supports these placeholders:
- `{url}` - the base URL
- `{z}` - zoom level
- `{x}` - tile X coordinate
- `{y}` - tile Y coordinate
- `{q}` - quadkey (automatically computed from x, y, z for Bing/Virtual Earth)

Default format: `{url}/{z}/{x}/{y}.png`

For tile servers with different URL structures:

```bash
# Standard XYZ tiles (OpenStreetMap)
uv run download_track_tiles.py https://tile.openstreetmap.org track.gpx output.mbtiles

# Tiles in a subdirectory
uv run download_track_tiles.py https://tiles.example-server.com/maps track.gpx output.mbtiles

# Google Satellite (tile type in base URL, query params in format)
uv run download_track_tiles.py --url-format "{url}&x={x}&y={y}&z={z}" \
  "https://mt1.google.com/vt/lyrs=s" track.gpx output.mbtiles

# Google Hybrid (different tile type in base URL, query params in format)
uv run download_track_tiles.py --url-format "{url}&x={x}&y={y}&z={z}" \
  "https://mt1.google.com/vt/lyrs=y" track.gpx output.mbtiles

# Virtual Earth / Bing Maps (uses quadkey)
uv run download_track_tiles.py --url-format "{url}/tiles/a{q}.jpeg?g=1" \
  "https://t0.ssl.ak.tiles.virtualearth.net" track.gpx output.mbtiles

# Custom tile server with path
uv run download_track_tiles.py "https://tiles.example.com/maps/terrain" track.gpx output.mbtiles
```

Note: Google Maps and Bing Maps have strict terms of service that generally prohibit bulk tile downloading. These examples are for educational purposes.

For legitimate offline usage, use providers that explicitly allow tile caching.

### Examples

Download tiles with 1km buffer, zoom 10-15:

```bash
uv run download_track_tiles.py --buffer 1000 --zoom 10 15 https://tile.server.com route.gpx maps.mbtiles
```

Add corner-based expansion for better centering:

```bash
uv run download_track_tiles.py --corner-expand https://tile.server.com route.gpx maps.mbtiles
```

Add 1 extra tile in all directions:

```bash
uv run download_track_tiles.py --tile-buffer 1 https://tile.server.com route.gpx maps.mbtiles
```

Combine multiple strategies (100m buffer around track, tiles at corner, 1 extra tile in all direction)

```bash
uv run download_track_tiles.py --buffer 100m --corner-expand --tile-buffer 1 https://tile.server.com route.gpx maps.mbtiles
```

Combine multiple strategies, add JPEG compression:

```bash
uv run download_track_tiles.py --buffer 500 --corner-expand --tile-buffer 1 --jpg 85 \
  https://tile.server.com route.gpx maps.mbtiles
```

## How it works

The downloader performs the following steps:

1. Parses GPX file and extracts track points
2. Creates oblique Mercator projection aligned with track direction
3. Buffers track by specified distance (accurate perpendicular buffering)
4. Calculates which tiles intersect the buffered area at each zoom level
5. Optionally applies different tile expansion strategies
6. Downloads tiles with retry logic and caching
7. Writes MBTiles database

The oblique Mercator projection minimizes distortion along the route and ensures accurate metric buffer calculation along the track.

### Tile expansion strategies

After the initial meter-based buffer shape computation and tile selection, you can apply additional tile expansion:

**Corner expansion (--corner-expand)**: For each track point, determines which quadrant of its tile the point falls into and downloads the 3 neighboring tiles that share that corner. This keeps your track better centered within the downloaded tile set. The expansion is adaptive based on where your actual track falls within each tile.

**Uniform buffer (--tile-buffer N)**: Adds N tiles in every direction around all base tiles. Simpler and more predictable than corner expansion, but downloads more tiles. Useful when you want guaranteed coverage beyond the meter-based buffer without regard to track position.

Both strategies can be combined. They build on top of the meter-based buffer, so you get accurate geometric buffering plus additional tile coverage.

## Merging tile databases

Use the included merge script to combine multiple MBTiles files:

```bash
uv run merge_tile_databases.py track1.mbtiles track2.mbtiles track3.mbtiles merged.mbtiles
```

The merge script handles duplicate tiles automatically and preserves metadata from the source files.

## Converting to OsmAnd format

OsmAnd uses a different SQLite schema than MBTiles. To convert your downloaded or merged tiles for use in OsmAnd, use the mbtiles2osmand tool:

```bash
# Install mbtiles2osmand from GitHub
pip install git+https://github.com/tarwirdur/mbtiles2osmand.git

# Convert single file
mbtiles2osmand output.mbtiles output.sqlitedb

# Convert merged file
mbtiles2osmand merged.mbtiles merged.sqlitedb
```

Copy the resulting `.sqlitedb` file to your OsmAnd tiles directory:

- Android: `/storage/emulated/0/Android/data/net.osmand/files/tiles/` or `/storage/emulated/0/Android/obb/net.osmand.plus/tiles/`
- iOS: Use iTunes file sharing to copy to OsmAnd's documents folder

The tiles will then appear in OsmAnd under Map source selection. You can overlay them on top of other maps or use them as a standalone base layer.

For more information, see https://github.com/tarwirdur/mbtiles2osmand

## Tile servers

Respect tile server usage policies. OpenStreetMap tile servers have usage limits and should not be used for bulk downloads. Consider using a commercial tile provider for large downloads or running your own tile server. Always check the tile server's terms of service before downloading.

The downloader includes a persistent cache in `~/.cache/track_tiles/` to avoid re-downloading tiles. Cache files are named by hostname (e.g., `tiles_tile_openstreetmap_org.cache`, `tiles_mt1_google_com.cache`) making them easy to identify and remove if needed.

## Dependencies

- gpxpy for GPX parsing
- shapely for geometry operations
- pyproj for coordinate transformations and projections
- mercantile for XYZ tile calculations
- requests for HTTP downloads
- Pillow for image format conversion
- numpy for numerical operations

## [License](LICENSE)

Use at your own risk. Be respectful of tile server resources.
