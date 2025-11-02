#!/usr/bin/env python3
"""
Track Tile Downloader - Download XYZ tiles along GPX track and save as MBTiles

Downloads map tiles from XYZ servers for GPS tracks with metric buffering and optional
expansion strategies. Outputs MBTiles format with format normalization and caching.

Architecture:
    - Pure functions for geometry and tile enumeration
    - Oblique Mercator projection for accurate perpendicular buffering
    - Composable tile expansion strategies (corner-based and uniform)
    - Per-hostname SQLite caching (~/.cache/track_tiles/)

Pipeline:
    1. Parse GPX → compute geodesic center and azimuth
    2. Project to oblique Mercator → buffer in meters → back to WGS84
    3. Enumerate intersecting tiles (not just bbox)
    4. Apply expansion strategies if requested
    5. Download with format detection/normalization
    6. Write MBTiles (TMS coordinates)

Buffer Width:
    The --buffer parameter creates a perpendicular buffer zone around the track in meters,
    computed in oblique Mercator projection for geometric accuracy. Only tiles whose
    polygons intersect this buffered corridor are downloaded (not all tiles in bbox).

Tile Expansion:
    --corner-expand: Add 3 neighbors sharing corner with track point (adaptive)
    --tile-buffer N: Add N tiles uniformly around each tile (predictable)

Coordinates:
    XYZ: Web tiles (top-left origin), TMS: MBTiles (bottom-left), Quadkey

Module Usage:
    Import and compose pure functions for custom tile processing. The architecture separates
    geometry processing, tile enumeration, and I/O into distinct phases. All functions use
    standard types (LineString, Transformer, Set[TileCoord]) with no hidden state.
    
    Example - custom pipeline with filtering:
    
        from pathlib import Path
        from download_track_tiles import (
            parse_gpx,
            create_track_transformers,
            coords_to_linestring_wgs84,
            transform_linestring_to_track_projection,
            buffer_track_metric,
            tiles_for_track,
            apply_corner_expansion,
        )
        
        # Parse and project track
        coords = parse_gpx(Path("track.gpx"))
        transformers = create_track_transformers(coords)
        line_wgs84 = coords_to_linestring_wgs84(coords)
        line_omerc = transform_linestring_to_track_projection(line_wgs84, transformers)
        
        # Buffer and enumerate tiles
        buffered = buffer_track_metric(line_omerc, 500)
        tiles = tiles_for_track(buffered, 8, 14)
        tiles = apply_corner_expansion(tiles, coords)
        
        # Custom filtering
        tiles = {(x, y, z) for x, y, z in tiles if z >= 10}
    
    See type aliases (LatLon, TileCoord) and docstrings for function contracts.

CLI Usage:
    # Basic - 500m buffer, zoom 4-16
    python download_track_tiles.py "https://tile.openstreetmap.org/{z}/{x}/{y}.png" \
           track.gpx output.mbtiles

    # Custom buffer and zoom range
    python download_track_tiles.py --buffer 1000 --zoom 8 14 <url> track.gpx out.mbtiles

    # With expansion strategies for better coverage
    python download_track_tiles.py --corner-expand --tile-buffer 1 <url> track.gpx out.mbtiles

    # Convert to JPEG for space savings
    python download_track_tiles.py --jpg 85 <url> track.gpx out.mbtiles
"""
import argparse
import io
import sqlite3
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from math import atan, pi, sinh
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import gpxpy
import mercantile
import numpy as np
import requests
from PIL import Image
from pyproj import Geod, Transformer
from shapely.geometry import LineString, box, shape
from shapely.ops import transform
from shapely.prepared import prep
from tqdm import tqdm

# Type aliases for clarity
LatLon = Tuple[float, float]  # WGS84 coordinates
WebMercator = Tuple[float, float]  # EPSG:3857 coordinates
TileCoord = Tuple[int, int, int]  # (x, y, z)


@dataclass(frozen=True)
class Config:
    """Immutable configuration for tile download"""

    gpx_path: Path
    base_url: str
    url_format: str
    output_file: Path
    buffer_meters: float
    zoom_min: int
    zoom_max: int
    jpeg_quality: Optional[int]
    force_overwrite: bool
    corner_expansion: bool  # Strategy 1: corner-based expansion
    tile_buffer: Optional[int]  # Strategy 2: uniform tile buffer


@dataclass(frozen=True)
class TrackTransformers:
    """Pair of transformers for track projection"""

    forward: Transformer  # WGS84 -> Oblique Mercator
    backward: Transformer  # Oblique Mercator -> WGS84


def detect_image_format(data: bytes) -> str:
    """Detect image format using Pillow"""
    im = Image.open(io.BytesIO(data))
    fmt = im.format.lower()

    # Normalize format names for metadata
    if fmt == "jpeg":
        return "jpg"
    elif fmt in ("png", "webp"):
        return fmt
    else:
        raise ValueError(f"Unsupported image format: {fmt}")


def estimate_jpeg_quality(data: bytes) -> Optional[int]:
    """
    Estimate JPEG quality from quantization tables using IJG scaling formula.
    Returns None if not JPEG or tables unavailable.
    """
    im = Image.open(io.BytesIO(data))
    
    if im.format != 'JPEG':
        return None
    
    qtables = im.quantization
    if not qtables or 0 not in qtables:
        return None
    
    actual = list(qtables[0])
    if len(actual) < 64:
        return None
    
    # IJG standard quantization table (quality 50 baseline)
    ijg_q50_luma = [
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99,
    ]
    
    # Reverse the formula: actual[i] = floor((S * base[i] + 50) / 100)
    # Approximate: S ≈ (actual[i] * 100) / base[i]
    ratios = []
    for i in range(64):
        if ijg_q50_luma[i] > 0:
            # Compute S percentage from: actual ≈ (S * base + 50) / 100
            s_estimate = (actual[i] * 100) / ijg_q50_luma[i]
            ratios.append(s_estimate)
    
    if not ratios:
        return None
    
    avg_s = sum(ratios) / len(ratios)
    
    # Reverse: if Q < 50: S = 5000/Q, else S = 200 - 2*Q
    if avg_s > 100:  # Q < 50
        quality = 5000 / avg_s
    else:  # Q >= 50
        quality = (200 - avg_s) / 2
    
    return int(round(max(1, min(100, quality))))


def normalize_tile_format(
    data: bytes, target_format: str, jpeg_quality: int = 85
) -> bytes:
    """Convert tile to target format if needed"""
    im = Image.open(io.BytesIO(data))
    current_format = im.format.lower()

    # Normalize format comparison
    if current_format == "jpeg":
        current_format = "jpg"

    if current_format == target_format:
        return data

    stream = io.BytesIO()

    if target_format == "jpg":
        im = im.convert("RGB")
        im.save(stream, format="JPEG", subsampling=0, quality=jpeg_quality)
    elif target_format == "png":
        im.save(stream, format="PNG", optimize=True)
    else:
        raise ValueError(f"Unsupported target format: {target_format}")

    return stream.getvalue()


def xyz_to_quadkey(x: int, y: int, z: int) -> str:
    """Convert XYZ tile coordinates to quadkey (for Bing/Virtual Earth)"""
    quadkey = []
    for i in range(z, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if x & mask:
            digit += 1
        if y & mask:
            digit += 2
        quadkey.append(str(digit))
    return "".join(quadkey)


# ============================================================================
# I/O Functions - Cached DB Connection
# ============================================================================


@lru_cache(maxsize=None)
def get_conn(db_path: Path) -> sqlite3.Connection:
    """Return a persistent SQLite connection for a given DB path"""
    conn = sqlite3.connect(db_path)
    return conn


# ============================================================================
# I/O Functions - Cached HTTP Downloads
# ============================================================================


@lru_cache(maxsize=None)
def get_cache_db_for_host(hostname: str) -> Path:
    """Get cache database path for hostname"""
    cache_dir = Path.home() / ".cache" / "track_tiles"
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_hostname = hostname.replace(".", "_")
    return cache_dir / f"tiles_{safe_hostname}.cache"


@lru_cache(maxsize=None)
def init_cache_db(db_path: Path) -> None:
    """Initialize cache database if it doesn't exist"""
    if db_path.exists():
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE tiles (url TEXT PRIMARY KEY, tile_data BLOB)")
    cur.execute("CREATE TABLE metadata (name TEXT, value TEXT)")
    conn.commit()
    conn.close()


def cached_requests_get(
    url: str, *, session: Optional[requests.Session] = None, **kwargs
) -> bytes:
    """
    Wrapper around requests.get() with caching.
    Returns tile data directly as bytes.
    """
    if session is None:
        session = requests.Session()

    from urllib.parse import urlparse

    parsed = urlparse(url)
    hostname = parsed.hostname or "unknown"

    db_path = get_cache_db_for_host(hostname)
    init_cache_db(db_path)

    # Check cache
    conn = get_conn(db_path)
    cur = conn.cursor()
    cur.execute("SELECT tile_data FROM tiles WHERE url=?", (url,))
    result = cur.fetchone()

    if result:
        return result[0]

    # Cache miss - make real request with session
    response = session.get(url, **kwargs)

    if response.status_code != 200:
        raise requests.HTTPError(f"HTTP {response.status_code}")

    # Cache successful response
    cur.execute(
        "INSERT OR REPLACE INTO tiles (url, tile_data) VALUES (?, ?)",
        (url, sqlite3.Binary(response.content)),
    )
    conn.commit()

    return response.content


# ============================================================================
# I/O Functions - HTTP Download
# ============================================================================


def download_single_tile(
    url_format: str,
    base_url: str,
    x: int,
    y: int,
    z: int,
    *,
    session: Optional[requests.Session] = None,
) -> bytes:
    """
    Download single tile with exponential backoff on server errors.
    Returns tile data as bytes.
    """
    max_retries = 5
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            # Compute quadkey for Virtual Earth / Bing Maps
            quadkey = xyz_to_quadkey(x, y, z)

            # Format URL with all available placeholders
            url = url_format.format(url=base_url, x=x, y=y, z=z, q=quadkey)

            return cached_requests_get(url, session=session, timeout=30)

        except requests.HTTPError as e:
            # Check if it's a server error (5xx)
            if "500" <= str(e).split()[1] < "600":
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    print(
                        f"  Server error for tile ({x},{y},{z}), "
                        f"retrying in {delay:.1f}s...",
                        file=sys.stderr,
                    )
                    time.sleep(delay)
                else:
                    raise
            else:
                # Client error - don't retry
                raise

        except requests.RequestException as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                print(
                    f"  Network error for tile ({x},{y},{z}): {e}, "
                    f"retrying in {delay:.1f}s...",
                    file=sys.stderr,
                )
                time.sleep(delay)
            else:
                raise

    raise requests.HTTPError(f"Failed to download tile after {max_retries} attempts")


def download_tiles(
    tiles: Set[TileCoord],
    url_format: str,
    base_url: str,
    *,
    session: Optional[requests.Session] = None,
    force_format: Optional[str] = None,
    jpeg_quality: int = 85,
) -> Tuple[Dict[TileCoord, bytes], str]:
    """Download all tiles and normalize format. Returns (tiles, format)"""
    tile_data = {}
    detected_format = None
    total = len(tiles)

    sorted_tiles = sorted(tiles, key=lambda t: (t[2], t[0], t[1]))

    with tqdm(total=total, desc="Downloading tiles", unit="tile") as pbar:
        for x, y, z in sorted_tiles:
            data = download_single_tile(url_format, base_url, x, y, z, session=session)

            # Detect format from first tile
            if detected_format is None:
                detected_format = detect_image_format(data)
                pbar.write(f"Detected tile format: {detected_format}")

                # Detect quality if JPEG
                if detected_format == "jpg":
                    detected_quality = estimate_jpeg_quality(data)
                    if detected_quality:
                        pbar.write(f"Detected JPEG quality: ~{detected_quality}")

            if force_format:
                data = normalize_tile_format(data, force_format, jpeg_quality)

            tile_data[(x, y, z)] = data
            pbar.update(1)

    final_format = force_format or detected_format
    return tile_data, final_format


# ============================================================================
# I/O Functions - MBTiles Database
# ============================================================================


def create_mbtiles(path: Path, metadata: Dict[str, str]) -> sqlite3.Connection:
    """Create MBTiles database with schema and metadata"""
    conn = sqlite3.connect(path)
    cur = conn.cursor()

    # Create tables
    cur.execute(
        """
        CREATE TABLE metadata (name TEXT, value TEXT)
    """
    )

    cur.execute(
        """
        CREATE TABLE tiles (
            zoom_level INTEGER,
            tile_column INTEGER,
            tile_row INTEGER,
            tile_data BLOB,
            PRIMARY KEY (zoom_level, tile_column, tile_row)
        )
    """
    )

    # Insert metadata
    for name, value in metadata.items():
        cur.execute("INSERT INTO metadata (name, value) VALUES (?, ?)", (name, value))

    conn.commit()
    return conn


def write_tiles_batch(conn: sqlite3.Connection, tiles: Dict[TileCoord, bytes]) -> None:
    """Write tiles to MBTiles database (TMS coordinates)"""
    cur = conn.cursor()

    for (x, y, z), data in tqdm(tiles.items(), desc="Writing tiles", unit="tile"):
        y_tms = (2**z - 1) - y
        cur.execute(
            "INSERT INTO tiles (zoom_level, tile_column, tile_row, tile_data) "
            "VALUES (?, ?, ?, ?)",
            (z, x, y_tms, sqlite3.Binary(data)),
        )

    conn.commit()


# ============================================================================
# Pure Functions - Tile Expansion Strategies
# ============================================================================


def get_corner_neighbors(
    x: int, y: int, z: int, lon: float, lat: float
) -> List[TileCoord]:
    """
    Get the 3 neighboring tiles that share a corner with (x,y,z) based on
    which quadrant the point (lon, lat) falls into within the tile.
    Returns list of 4 tiles total (original + 3 neighbors).
    """
    # Convert tile coordinates to lat/lon bounds
    n = 2.0**z

    # Tile bounds
    tile_lon_min = x / n * 360.0 - 180.0
    tile_lon_max = (x + 1) / n * 360.0 - 180.0

    lat_rad_min = atan(sinh(pi * (1 - 2 * (y + 1) / n)))
    lat_rad_max = atan(sinh(pi * (1 - 2 * y / n)))
    tile_lat_min = lat_rad_min * 180.0 / pi
    tile_lat_max = lat_rad_max * 180.0 / pi

    # Determine which subtile quadrant the point is in
    lon_mid = (tile_lon_min + tile_lon_max) / 2
    lat_mid = (tile_lat_min + tile_lat_max) / 2

    # Determine offsets based on quadrant
    dx = 1 if lon >= lon_mid else -1
    dy = 1 if lat <= lat_mid else -1  # Note: tile y increases downward

    # Generate 4 tiles around the corner
    tiles = [
        (x, y, z),  # Original
        (x + dx, y, z),  # Horizontal neighbor
        (x, y + dy, z),  # Vertical neighbor
        (x + dx, y + dy, z),  # Diagonal neighbor
    ]

    # Filter out invalid tiles (negative coords or beyond bounds)
    max_tile = 2**z - 1
    return [
        (tx, ty, tz)
        for tx, ty, tz in tiles
        if 0 <= tx <= max_tile and 0 <= ty <= max_tile
    ]


def apply_corner_expansion(
    tiles: Set[TileCoord], coords_wgs84: List[LatLon]
) -> Set[TileCoord]:
    """
    Apply corner-based expansion (Strategy 1).
    For each track point, finds its tile and adds 3 neighbors sharing the corner.

    Args:
        tiles: Base set of tiles from meter-based buffer
        coords_wgs84: Original track points as (lon, lat)

    Returns:
        Expanded set of tiles
    """
    expanded = set(tiles)

    # For each track point, add corner neighbors
    for lon, lat in coords_wgs84:
        for z in {t[2] for t in tiles}:  # Get unique zoom levels
            # Find which tile this point is in
            tile = mercantile.tile(lon, lat, z)
            corner_tiles = get_corner_neighbors(tile.x, tile.y, z, lon, lat)
            expanded.update(corner_tiles)

    return expanded


def apply_tile_buffer_expansion(
    tiles: Set[TileCoord], buffer_size: int
) -> Set[TileCoord]:
    """
    Apply uniform tile buffer expansion (Strategy 2).
    Adds N tiles in each direction around every tile.

    Args:
        tiles: Base set of tiles from meter-based buffer
        buffer_size: Number of tiles to buffer in each direction

    Returns:
        Expanded set of tiles
    """
    if buffer_size <= 0:
        return tiles

    expanded = set()

    for x, y, z in tiles:
        max_tile = 2**z - 1

        # Add all tiles in buffer radius
        for dx in range(-buffer_size, buffer_size + 1):
            for dy in range(-buffer_size, buffer_size + 1):
                tx, ty = x + dx, y + dy

                # Keep only valid tiles
                if 0 <= tx <= max_tile and 0 <= ty <= max_tile:
                    expanded.add((tx, ty, z))

    return expanded


# ============================================================================
# Pure Functions - GPX and Geometry Processing
# ============================================================================


def parse_gpx(filepath: Path) -> List[LatLon]:
    """Parse GPX file and extract all track points as (lon, lat) tuples"""
    with open(filepath, "r") as f:
        gpx = gpxpy.parse(f)

    coords = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                coords.append((point.longitude, point.latitude))

    if not coords:
        raise ValueError("No track points found in GPX file")

    return coords


def coords_to_linestring_wgs84(coords: List[LatLon]) -> LineString:
    """Convert coordinate list to Shapely LineString in WGS84"""
    return LineString(coords)


def ecef_to_geodetic_lat(geod: Geod, hyp, z, max_iter=5):
    """
    Convert ECEF coordinates to geodetic latitude using parametric latitude.

    Parameters
    ----------
    hyp : float
        Distance from Z-axis, sqrt(x^2 + y^2)
    z : float
        Z coordinate
    geod : pyproj.Geod
        Ellipsoid object
    max_iter : int
        Maximum iterations

    Returns
    -------
    phi : float
        Geodetic latitude in radians
    """
    a = geod.a
    f = geod.f
    b = a * (1 - f)
    e2 = 2 * f - f**2

    # Initial guess: parametric latitude
    phi = np.arctan2((b / a) * z, hyp)

    for _ in range(max_iter):
        N_phi = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
        phi = np.arctan2(z + e2 * N_phi * np.sin(phi), hyp)

    return phi


def geodesic_center_fancy(geod: Geod, lons: np.ndarray, lats: np.ndarray) -> LatLon:
    """
    Compute the geodesic projection of the mean of points on an ellipsoid.

    Parameters
    ----------
    lons : array-like
        Longitudes in degrees
    lats : array-like
        Geodetic latitudes in degrees
    geod : pyproj.Geod
        Ellipsoid, e.g., Geod(ellps="WGS84")

    Returns
    -------
    mean_lon, mean_lat : float
        Geodesic mean longitude and latitude in degrees
    """
    # Convert input to radians
    lons_rad = np.radians(lons)
    lats_rad = np.radians(lats)

    # Ellipsoid parameters
    a = geod.a
    f = geod.f
    b = a * (1 - f)
    e2 = 2 * f - f**2

    # Convert geodetic to ECEF
    cos_lat = np.cos(lats_rad)
    sin_lat = np.sin(lats_rad)
    N = a / np.sqrt(1 - e2 * sin_lat**2)
    x = N * cos_lat * np.cos(lons_rad)
    y = N * cos_lat * np.sin(lons_rad)
    z = N * (1 - e2) * sin_lat

    # Mean in 3D space
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    z_mean = np.mean(z)

    # Mean longitude
    mean_lon = np.degrees(np.arctan2(y_mean, x_mean))

    # Compute geodetic latitude
    hyp_mean = np.sqrt(x_mean**2 + y_mean**2)
    mean_lat = np.degrees(ecef_to_geodetic_lat(geod, hyp_mean, z_mean))

    return mean_lon, mean_lat


def geodesic_center_sphere(lons: np.ndarray, lats: np.ndarray) -> LatLon:
    """Compute geometric center on ellipsoid via 3D Cartesian coordinates"""

    # Convert to radians
    lons_rad = np.radians(lons)
    lats_rad = np.radians(lats)

    # Convert to 3D Cartesian (unit sphere approximation)
    x = np.cos(lats_rad) * np.cos(lons_rad)
    y = np.cos(lats_rad) * np.sin(lons_rad)
    z = np.sin(lats_rad)

    # Mean in 3D space
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    z_mean = np.mean(z)

    # Convert back to lat/lon
    center_lon = np.degrees(np.arctan2(y_mean, x_mean))
    center_lat = np.degrees(np.arctan2(z_mean, np.sqrt(x_mean**2 + y_mean**2)))

    return center_lon, center_lat


def create_track_transformers(coords_wgs84: List[LatLon]) -> TrackTransformers:
    # Compute proper geodesic center
    lons, lats = np.array(coords_wgs84, dtype=float).T

    # Reference Ellipsoid
    geod = Geod(ellps="WGS84")

    # Compute the geodesic projection of the mean of all points
    center_lon, center_lat = geodesic_center_fancy(geod, lons, lats)

    # Azimuth and distance from center to each point

    # Vectorized azimuth calculation using pyproj.Geod
    geod = Geod(ellps="WGS84")
    segment_azimuths, _, segment_lengths = geod.inv(
        lons[:-1], lats[:-1], lons[1:], lats[1:]
    )

    # Weight by distance
    azimuths_rad = np.radians(segment_azimuths)
    sin_weighted = np.sum(segment_lengths * np.sin(azimuths_rad))
    cos_weighted = np.sum(segment_lengths * np.cos(azimuths_rad))
    azimuth = np.degrees(np.arctan2(sin_weighted, cos_weighted))

    # Build the oblique Mercator projection string
    print(f"  Oblique Mercator, center: {center_lon}, {center_lat}, azimuth: {azimuth}")
    omerc_proj = (
        f"+proj=omerc +lat_0={center_lat} +lonc={center_lon} "
        f"+alpha={azimuth} +datum=WGS84 +units=m"
    )

    forward = Transformer.from_crs("EPSG:4326", omerc_proj, always_xy=True)
    backward = Transformer.from_crs(omerc_proj, "EPSG:4326", always_xy=True)

    return TrackTransformers(forward=forward, backward=backward)


def transform_linestring_to_track_projection(
    line_wgs84: LineString, transformers: TrackTransformers
) -> LineString:
    """Transform LineString from WGS84 to oblique Mercator for the track"""
    coords_omerc = [
        transformers.forward.transform(lon, lat) for lon, lat in line_wgs84.coords
    ]
    return LineString(coords_omerc)


def buffer_track_metric(line_omerc: LineString, distance_meters: float):
    """Buffer track in oblique Mercator projection (true meters perpendicular to route)"""
    return line_omerc.buffer(distance_meters)


def get_bbox_wgs84_from_track_projection(
    polygon_omerc, transformers: TrackTransformers
) -> Tuple[float, float, float, float]:
    """Get bounding box in WGS84 from oblique Mercator polygon"""
    minx, miny, maxx, maxy = polygon_omerc.bounds

    # Transform all four corners
    corners_omerc = [
        (minx, miny),
        (minx, maxy),
        (maxx, miny),
        (maxx, maxy),
    ]

    corners_wgs84 = [transformers.backward.transform(x, y) for x, y in corners_omerc]
    lons, lats = zip(*corners_wgs84)

    west, south, east, north = min(lons), min(lats), max(lons), max(lats)
    return west, south, east, north


# ============================================================================
# Pure Functions - Tile Enumeration
# ============================================================================


def tiles_for_bbox(
    bbox_wgs84: Tuple[float, float, float, float], zoom: int
) -> Set[TileCoord]:
    """Get all tile coordinates that intersect bbox at given zoom level"""
    west, south, east, north = bbox_wgs84
    tiles = mercantile.tiles(west, south, east, north, zooms=zoom)
    return {(tile.x, tile.y, tile.z) for tile in tiles}


def all_tiles_for_bbox(
    bbox_wgs84: Tuple[float, float, float, float], zoom_min: int, zoom_max: int
) -> Set[TileCoord]:
    """Get all tiles across zoom range"""
    all_tiles = set()
    for zoom in range(zoom_min, zoom_max + 1):
        all_tiles.update(tiles_for_bbox(bbox_wgs84, zoom))
    return all_tiles


def tiles_for_track(polygon_wgs84, zoom_min, zoom_max) -> Set[TileCoord]:
    """Return only tiles intersecting the buffered track polygon (optimized)."""
    tiles = set()
    west, south, east, north = polygon_wgs84.bounds
    prepared = prep(polygon_wgs84)

    for z in range(zoom_min, zoom_max + 1):
        for tile in mercantile.tiles(west, south, east, north, zooms=z):
            tile_poly = box(*mercantile.bounds(tile))
            if prepared.intersects(tile_poly):
                tiles.add((tile.x, tile.y, tile.z))
    return tiles


# ============================================================================
# Main Pipeline
# ============================================================================


def process_track(config: Config) -> None:
    """Main processing pipeline"""
    print(f"Processing GPX track: {config.gpx_path}")

    # 1. Parse and process track geometry
    coords_wgs84 = parse_gpx(config.gpx_path)
    print(f"  Found {len(coords_wgs84)} track points")
    print(f"  Requested buffer zone: {config.buffer_meters}m")

    line_wgs84 = coords_to_linestring_wgs84(coords_wgs84)
    transformers = create_track_transformers(coords_wgs84)
    line_omerc = transform_linestring_to_track_projection(line_wgs84, transformers)
    track_buffer_omerc = buffer_track_metric(line_omerc, config.buffer_meters)
    track_buffer_wgs84 = transform(transformers.backward.transform, track_buffer_omerc)
    bbox_wgs84 = get_bbox_wgs84_from_track_projection(track_buffer_omerc, transformers)

    print(f"  BBox (WGS84): {bbox_wgs84}")

    # 2. Enumerate tiles from meter-based buffer (original logic)
    tiles = tiles_for_track(track_buffer_wgs84, config.zoom_min, config.zoom_max)
    print(f"  Base tiles (from {config.buffer_meters}m buffer): {len(tiles)}")

    # 3. Apply tile expansion strategies if requested
    if config.corner_expansion:
        tiles = apply_corner_expansion(tiles, coords_wgs84)
        print(f"  After corner expansion: {len(tiles)} tiles")

    if config.tile_buffer is not None and config.tile_buffer > 0:
        tiles = apply_tile_buffer_expansion(tiles, config.tile_buffer)
        print(f"  After {config.tile_buffer}-tile buffer: {len(tiles)} tiles")

    print(
        f"  Final tiles to download: {len(tiles)} (zoom {config.zoom_min}-{config.zoom_max})"
    )

    # 4. Download tiles with format detection/normalization using a shared session
    force_format = "jpg" if config.jpeg_quality else None
    with requests.Session() as session:
        tile_data, final_format = download_tiles(
            tiles,
            config.url_format,
            config.base_url,
            session=session,
            force_format=force_format,
            jpeg_quality=config.jpeg_quality or 85,
        )
        print(f"  Successfully downloaded: {len(tile_data)} tiles ({final_format} format)")

    if not tile_data:
        print("No tiles downloaded, aborting.", file=sys.stderr)
        sys.exit(1)

    # 5. Write MBTiles
    mbtiles_path = config.output_file

    if mbtiles_path.exists():
        if not config.force_overwrite:
            print(
                f"Error: {mbtiles_path} already exists. Use -f to overwrite.",
                file=sys.stderr,
            )
            sys.exit(1)
        else:
            # Remove existing file before creating new one
            mbtiles_path.unlink()
            print(f"  Removed existing {mbtiles_path}")

    metadata = {
        "name": config.output_file.stem,
        "type": "baselayer",
        "version": "1.0",
        "description": f"Tiles along GPX track with {config.buffer_meters}m buffer",
        "format": final_format,
        "bounds": ",".join(map(str, bbox_wgs84)),
        "minzoom": str(config.zoom_min),
        "maxzoom": str(config.zoom_max),
    }

    print(f"Writing MBTiles: {mbtiles_path}")
    conn = create_mbtiles(mbtiles_path, metadata)
    write_tiles_batch(conn, tile_data)
    conn.close()

    print(f"\nComplete!")
    print(f"  MBTiles: {mbtiles_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download XYZ tiles along GPX track and save as MBTiles/OsmAnd",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download tiles with 500m buffer, zoom 4-16
  %(prog)s "https://tile.openstreetmap.org/{z}/{x}/{y}.png" track.gpx output.mbtiles
  
  # With custom buffer and zoom range
  %(prog)s --buffer 1000 --zoom 8 16 "https://tile.server/{z}/{x}/{y}.png" track.gpx output.tiles
  
  # Add corner-based expansion (strategy 1)
  %(prog)s --corner-expand "https://tile.server/{z}/{x}/{y}.png" track.gpx output.tiles
  
  # Add 2-tile uniform buffer (strategy 2)
  %(prog)s --tile-buffer 2 "https://tile.server/{z}/{x}/{y}.png" track.gpx output.tiles
  
  # Combine both expansion strategies
  %(prog)s --corner-expand --tile-buffer 1 "https://tile.server/{z}/{x}/{y}.png" track.gpx output
  
  # Force convert to JPEG (quality 85) to save space
  %(prog)s output --jpg 85 "https://tile.server/{z}/{x}/{y}.png" track.gpx
        """,
    )

    parser.add_argument(
        "-b",
        "--buffer",
        type=float,
        default=500.0,
        help="Buffer distance from track in meters (default: 500)",
    )
    parser.add_argument(
        "-z",
        "--zoom",
        nargs=2,
        type=int,
        metavar=("MIN", "MAX"),
        default=[4, 16],
        help="Zoom level range (default: 4 16)",
    )
    parser.add_argument(
        "--corner-expand",
        action="store_true",
        help="Add corner-based tile expansion (strategy 1: adaptive to track position)",
    )
    parser.add_argument(
        "--tile-buffer",
        type=int,
        metavar="N",
        help="Add N-tile uniform buffer around all tiles (strategy 2)",
    )
    parser.add_argument(
        "--jpg",
        type=int,
        metavar="QUALITY",
        help="Convert tiles to JPEG with quality 1-100 (default: keep original format)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite output files if they exist",
    )
    parser.add_argument(
        "--url-format",
        default="{url}/{z}/{x}/{y}.png",
        help="URL format string. Placeholders: {url}, {x}, {y}, {z}, {q} (quadkey). Default: '{url}/{z}/{x}/{y}.png'",
    )

    parser.add_argument("base_url", help="XYZ tile base URL")
    parser.add_argument("gpx_path", type=Path, help="Input GPX file")
    parser.add_argument(
        "output_file", type=Path, help="Output filename (with extension)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.gpx_path.exists():
        print(f"Error: GPX file not found: {args.gpx_path}", file=sys.stderr)
        sys.exit(1)

    if args.jpg is not None and not (1 <= args.jpg <= 100):
        print("Error: JPEG quality must be between 1 and 100", file=sys.stderr)
        sys.exit(1)

    if not (0 <= args.zoom[0] <= args.zoom[1] <= 20):
        print("Error: Invalid zoom range (must be 0-20, min <= max)", file=sys.stderr)
        sys.exit(1)

    if args.tile_buffer is not None and args.tile_buffer < 0:
        print("Error: Tile buffer must be non-negative", file=sys.stderr)
        sys.exit(1)

    # Create config
    config = Config(
        url_format=args.url_format,
        base_url=args.base_url,
        gpx_path=args.gpx_path,
        output_file=args.output_file,
        buffer_meters=args.buffer,
        zoom_min=args.zoom[0],
        zoom_max=args.zoom[1],
        jpeg_quality=args.jpg,
        force_overwrite=args.force,
        corner_expansion=args.corner_expand,
        tile_buffer=args.tile_buffer,
    )

    # Run pipeline
    process_track(config)


if __name__ == "__main__":
    main()
