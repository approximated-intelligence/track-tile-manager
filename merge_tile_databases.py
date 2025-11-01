#!/usr/bin/env python3
"""
Track Tile Merge - Combine multiple MBTiles databases into one
"""
import argparse
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def get_mbtiles_stats(conn: sqlite3.Connection) -> dict:
    """Get statistics from an open MBTiles connection"""
    cur = conn.cursor()

    # Get metadata
    metadata = {
        row[0]: row[1] for row in cur.execute("SELECT name, value FROM metadata")
    }

    # Get zoom range
    cur.execute("SELECT MIN(zoom_level), MAX(zoom_level) FROM tiles")
    min_zoom, max_zoom = cur.fetchone()

    # Get tile count
    cur.execute("SELECT COUNT(*) FROM tiles")
    tile_count = cur.fetchone()[0]

    # Get bounds if available
    bounds = metadata.get("bounds", None)
    if bounds:
        bounds = tuple(map(float, bounds.split(",")))

    return {
        "format": "mbtiles",
        "tile_count": tile_count,
        "min_zoom": min_zoom,
        "max_zoom": max_zoom,
        "bounds": bounds,
        "metadata": metadata,
    }


def merge_bounds(
    bounds_list: List[Optional[Tuple[float, float, float, float]]],
) -> Optional[Tuple[float, float, float, float]]:
    """Merge multiple bounding boxes into one (west, south, east, north)"""
    bounds_list = [b for b in bounds_list if b is not None]
    if not bounds_list:
        return None

    west = min(b[0] for b in bounds_list)
    south = min(b[1] for b in bounds_list)
    east = max(b[2] for b in bounds_list)
    north = max(b[3] for b in bounds_list)
    return west, south, east, north


def create_output_mbtiles(path: Path, metadata: dict) -> sqlite3.Connection:
    """Create output MBTiles database"""
    conn = sqlite3.connect(path)
    cur = conn.cursor()

    # Create schema
    cur.execute("CREATE TABLE metadata (name TEXT, value TEXT)")
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


def copy_mbtiles_tiles(
    source_conn: sqlite3.Connection, dest_conn: sqlite3.Connection
) -> Tuple[int, int]:
    """Copy tiles from an open MBTiles source connection to destination"""
    scur = source_conn.cursor()
    dcur = dest_conn.cursor()

    copied = 0
    skipped = 0

    for z, x, y, data in scur.execute(
        "SELECT zoom_level, tile_column, tile_row, tile_data FROM tiles"
    ):
        try:
            dcur.execute(
                "INSERT INTO tiles (zoom_level, tile_column, tile_row, tile_data) VALUES (?, ?, ?, ?)",
                (z, x, y, sqlite3.Binary(data)),
            )
            copied += 1
        except sqlite3.IntegrityError:
            # Tile already exists, skip
            skipped += 1

    dest_conn.commit()
    return copied, skipped


def merge_mbtiles(output_path: Path, input_paths: List[Path], force: bool) -> None:
    """Merge multiple MBTiles files into one"""
    if output_path.exists() and not force:
        print(
            f"Error: {output_path} already exists. Use -f to overwrite.",
            file=sys.stderr,
        )
        sys.exit(1)
    if output_path.exists():
        output_path.unlink()

    print("Opening input files...")
    input_conns = [sqlite3.connect(f"file:{p}?mode=ro", uri=True) for p in input_paths]

    print("Analyzing input files...")
    stats_list = [get_mbtiles_stats(c) for c in input_conns]
    for path, stats in zip(input_paths, stats_list):
        print(
            f"  {path.name}: {stats['tile_count']} tiles, zoom {stats['min_zoom']}-{stats['max_zoom']}"
        )

    # Compute merged metadata
    all_bounds = [s["bounds"] for s in stats_list]
    merged_bounds = merge_bounds(all_bounds)
    min_zoom = min(s["min_zoom"] for s in stats_list)
    max_zoom = max(s["max_zoom"] for s in stats_list)

    base_metadata = stats_list[0]["metadata"].copy()
    base_metadata.update(
        {
            "name": output_path.stem + "_merged",
            "description": f"Merged from {len(input_paths)} tile databases",
            "minzoom": str(min_zoom),
            "maxzoom": str(max_zoom),
        }
    )
    if merged_bounds:
        base_metadata["bounds"] = ",".join(map(str, merged_bounds))

    # Create output database
    print(f"\nCreating {output_path}...")
    dest_conn = create_output_mbtiles(output_path, base_metadata)

    total_copied = 0
    total_skipped = 0

    for path, src_conn in zip(input_paths, input_conns):
        print(f"  Merging {path.name}...")
        copied, skipped = copy_mbtiles_tiles(src_conn, dest_conn)
        total_copied += copied
        total_skipped += skipped
        print(f"    Copied: {copied}, Skipped: {skipped} (duplicates)")

    dest_conn.close()
    for conn in input_conns:
        conn.close()

    print(f"\nMerge complete!")
    print(f"  Total tiles: {total_copied}")
    print(f"  Duplicates skipped: {total_skipped}")
    print(f"  Zoom range: {min_zoom}-{max_zoom}")
    if merged_bounds:
        print(f"  Bounds: {merged_bounds}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple MBTiles databases into one",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge MBTiles files
  %(prog)s track1.mbtiles track2.mbtiles track3.mbtiles output.mbtiles
  
  # Force overwrite existing output
  %(prog)s -f track1.mbtiles track2.mbtiles output.mbtiles
        """,
    )

    parser.add_argument(
        "-f", "--force", action="store_true", help="Overwrite output file if it exists"
    )

    parser.add_argument("inputs", nargs="+", type=Path, help="Input files to merge")
    parser.add_argument("output", type=Path, help="Output file (.mbtiles or .sqlitedb)")

    args = parser.parse_args()

    # Validate number of input arguments
    if len(args.inputs) < 2:
        print("Error: Need at least 2 input files to merge", file=sys.stderr)
        sys.exit(1)

    # Validate inputs exist
    for path in args.inputs:
        if not path.exists():
            print(f"Error: Input file not found: {path}", file=sys.stderr)
            sys.exit(1)

    merge_mbtiles(args.output, args.inputs, args.force)


if __name__ == "__main__":
    main()
