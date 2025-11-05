import utils.pathfinding as pathfinding
import copy
import logging
import sys
from typing import List, Tuple, Dict, Optional, Set
from pokemon_env.enums import MetatileBehavior

# Enable logging to see pathfinder messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Based on the logic in pathfinding.py
FLOOR = (1, 0, 0, 0)
WALL = (2, 1, 1, 0)   # behavior 1 = impassable, collision 1 = blocked
LEDGE_S = (3, MetatileBehavior.JUMP_SOUTH.value, 0, 0)
LEDGE_E = (4, MetatileBehavior.JUMP_EAST.value, 0, 0)

PRINT_LEGEND = {
    FLOOR: ".",   # Walkable floor
    WALL: "#",    # Impassable wall
    LEDGE_S: "v", # Ledge (Jump South)
    LEDGE_E: ">", # Ledge (Jump East)
}

# --- 15x15 Base Map ---
logger.info("Creating 15x15 base map...")

# Start with a 15x15 grid of FLOOR
base_map_tiles = [[FLOOR for _ in range(15)] for _ in range(15)]

# Add border walls
for i in range(15):
    base_map_tiles[0][i] = WALL  # Top row
    base_map_tiles[14][i] = WALL # Bottom row
    base_map_tiles[i][0] = WALL  # Left column
    base_map_tiles[i][14] = WALL # Right column

# Add a horizontal ledge at y=7
# This ledge only allows jumping SOUTH (down)
for x in range(1, 14):
    base_map_tiles[7][x] = LEDGE_S
    
# Add a vertical wall at x=10, from y=8 to y=13
# This is to test basic wall avoidance
for y in range(8, 14):
    base_map_tiles[y][10] = WALL

logger.info("Base map created.")

def print_map_layout(
    test_map_tiles, 
    start: Tuple[int, int], 
    goal: Tuple[int, int], 
    npcs: List[Tuple[int, int]]
):
    """
    Prints a visual grid of the map layout with Start, Goal, and NPCs.
    """
    # Create a grid of characters from the tile data
    grid = []
    for y, row in enumerate(test_map_tiles):
        char_row = []
        for tile_tuple in row:
            # Look up the character, default to '?' if unknown
            char = PRINT_LEGEND.get(tile_tuple, "?")
            char_row.append(char)
        grid.append(char_row)

    # Draw NPCs
    for (x, y) in npcs:
        if 0 <= y < len(grid) and 0 <= x < len(grid[y]):
            grid[y][x] = "N"

    # Draw Start and Goal
    if 0 <= start[1] < len(grid) and 0 <= start[0] < len(grid[start[1]]):
        grid[start[1]][start[0]] = "S"
    
    if 0 <= goal[1] < len(grid) and 0 <= goal[0] < len(grid[goal[1]]):
        grid[goal[1]][goal[0]] = "G"

    # Print the final grid
    print("Map Layout:")
    for y, row in enumerate(grid):
        # Add row numbers for clarity
        print(f"{y:2d} " + "".join(row))
    print("-" * 30)

def run_test(test_name, start, goal, npcs, custom_tiles=None):
    """
    Runs a single pathfinding test with a given map setup.
    """
    print(f"\n--- {test_name} ---")
    print(f"Start: {start}, Goal: {goal}, NPCs: {npcs}")
    
    # Create a deep copy of the map so tests don't interfere
    test_map_tiles = copy.deepcopy(base_map_tiles)
    
    # Add any custom tiles for this specific test
    if custom_tiles:
        for (x, y, tile) in custom_tiles:
            if 0 <= y < 15 and 0 <= x < 15:
                test_map_tiles[y][x] = tile

    # --- NEW: Print the map layout ---
    print_map_layout(test_map_tiles, start, goal, npcs)
    
    # Create the map_data and game_state
    map_data = {
        'width': 15,
        'height': 15,
        'tiles': test_map_tiles
    }
    
    game_state = {
        'map_data': map_data,
        'npcs': [{'x': npc[0], 'y': npc[1]} for npc in npcs]
    }

    # Instantiate the Pathfinder from the imported module
    pf = pathfinding.Pathfinder()
    
    # Find the path (this is the original version)
    path_buttons = pf.find_path(start, goal, game_state, max_distance=100)
    
    if path_buttons:
        print(f"Result: Path found! ({len(path_buttons)} steps)")
        print(f"Buttons: {path_buttons}")
    else:
        print("Result: No path found.")
    print("-" * 30)

# --- Define and Run Tests ---

# Test 1: Cross ledge successfully (downwards)
# Start above the ledge, goal is below the ledge.
run_test(
    "TEST 1: JUMP DOWN LEDGE (SUCCESS)",
    start=(5, 5),
    goal=(5, 9),
    npcs=[]
)

# Test 2: Fail to cross ledge (upwards)
# Start below the ledge, goal is above. This should be impossible.
run_test(
    "TEST 2: JUMP UP LEDGE (FAIL)",
    start=(5, 9),
    goal=(5, 5),
    npcs=[]
)

# Test 3: Path around a simple NPC
# NPC is blocking the direct horizontal path.
run_test(
    "TEST 3: PATH AROUND NPC",
    start=(2, 2),
    goal=(8, 2),
    npcs=[(5, 2)] # NPC at (5, 2)
)

# Test 4: Path around a wall
# Path is blocked by the vertical wall at x=10.
run_test(
    "TEST 4: PATH AROUND WALL",
    start=(8, 12),
    goal=(12, 12),
    npcs=[]
)

# Test 5: Ledge jump blocked by an NPC
# Path requires jumping the ledge, but an NPC is on the landing spot.
run_test(
    "TEST 5: LEDGE JUMP BLOCKED BY NPC (FAIL)",
    start=(5, 5),
    goal=(5, 9),
    npcs=[(5, 7)] # NPC is ON the ledge tile
)

# Test 6: A more complex path
# Requires navigating around the wall AND jumping the ledge.
run_test(
    "TEST 6: COMPLEX PATH (WALL + LEDGE)",
    start=(8, 5), # Above ledge, left of wall
    goal=(12, 12), # Below ledge, right of wall
    npcs=[]
)

print("\nTest suite complete.")

def run_custom_map_test(test_name, start, goal, npcs, map_layout, tile_legend):
    """
    Runs a pathfinding test on a custom, string-defined map.
    """
    print(f"\n--- {test_name} (CUSTOM MAP) ---")
    print(f"Start: {start}, Goal: {goal}, NPCs: {npcs}")

    # --- NEW: Print the custom map layout first ---
    # Convert map_layout strings to a 2D list of characters
    char_grid = [list(row) for row in map_layout]
    
    # Draw NPCs
    for (x, y) in npcs:
        if 0 <= y < len(char_grid) and 0 <= x < len(char_grid[y]):
            char_grid[y][x] = "N"
            
    # Redraw S and G just in case NPC overwrote them
    char_grid[start[1]][start[0]] = "S"
    char_grid[goal[1]][goal[0]] = "G"

    print("Map Layout:")
    for y, row in enumerate(char_grid):
        print(f"{y:2d} " + "".join(row))
    print("-" * 30)

    # --- Now, proceed with the pathfinding logic ---
    
    # Convert the string map into the tile tuple map
    test_map_tiles = []
    try:
        for y, row_str in enumerate(map_layout):
            row_tiles = []
            for x, char in enumerate(row_str):
                # Fix for S/G not being in legend
                if char not in tile_legend:
                    if char in ('S', 'G'): # Treat S and G as FLOOR
                        row_tiles.append(tile_legend[' ']) 
                    else:
                        print(f"ERROR: Character '{char}' at ({x},{y}) not in tile_legend!")
                        return
                else:
                    row_tiles.append(tile_legend[char])
            test_map_tiles.append(row_tiles)
    except Exception as e:
        print(f"Error building map: {e}")
        return

    # Get map dimensions
    map_height = len(test_map_tiles)
    map_width = len(test_map_tiles[0]) if map_height > 0 else 0

    # Create the map_data and game_state
    map_data = {
        'width': map_width,
        'height': map_height,
        'tiles': test_map_tiles
    }
    
    game_state = {
        'map_data': map_data,
        'npcs': [{'x': npc[0], 'y': npc[1]} for npc in npcs]
    }

    # Instantiate and run the pathfinder
    pf = pathfinding.Pathfinder()
    path_buttons = pf.find_path(start, goal, game_state, max_distance=100)
    
    if path_buttons:
        print(f"Result: Path found! ({len(path_buttons)} steps)")
        print(f"Buttons: {path_buttons}")
    else:
        print("Result: No path found.")
    print("-" * 30)