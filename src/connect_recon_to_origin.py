from typing import Dict, List, Tuple
import split
from arc.species.converter import xyz_to_dmat

def get_custom_adjlist(
    adjlist: Dict[int, List[int]],
    symbols: Tuple[str, ...],
    heavy_atoms_only: bool = True,
) -> Dict[int, List[List[int]]]:
    """
    Create a custom adjacency list for heavy atoms with nested paths.

    Args:
        adjlist (Dict[int, List[int]]): The input adjacency list.
        symbols (Tuple[str]): The chemical symbols of the atoms.
        heavy_atoms_only (bool): Whether to include only heavy atoms.

    Returns:
        Dict[int, List[List[int]]]: A custom adjacency list with nested paths.
    """
    def is_heavy(atom: int) -> bool:
        """Check if an atom is a heavy atom (non-hydrogen)."""
        return symbols[atom] != "H"

    def traverse_path(start_atom: int, current_atom: int, visited: set) -> List[int]:
        """
        Recursively traverse paths from the current atom.

        Args:
            start_atom (int): The original atom where the path started.
            current_atom (int): The atom currently being traversed.
            visited (set): The set of atoms already visited in this traversal.

        Returns:
            List[int]: The traversed path.
        """
        path = [current_atom]
        visited.add(current_atom)
        for neighbor in adjlist[current_atom]:
            if neighbor == start_atom or neighbor in visited:
                continue
            if is_heavy(neighbor):
                path.extend(traverse_path(start_atom, neighbor, visited))
        return path

    custom_adjlist = {}
    for atom, neighbors in adjlist.items():
        if heavy_atoms_only and not is_heavy(atom):
            continue

        # Initialize paths for this heavy atom
        custom_adjlist[atom] = []
        visited_neighbors = set()
        for neighbor in neighbors:
            if is_heavy(neighbor) and neighbor not in visited_neighbors:
                # Traverse paths starting with this heavy neighbor
                path = traverse_path(atom, neighbor, visited_neighbors)
                custom_adjlist[atom].append(path)

    return custom_adjlist

def flatten(nested_list):
    return [item for sublist in nested_list for item in sublist] if isinstance(nested_list, list) else nested_list

def get_adjacent_matches(recon_key, recon_symbols, recon_adjlist, matched_original_keys, orig_symbols, original_adjlist, include_non_heavy=False):
    def match_paths_at_depth(depth, match_list):
        adjacent_matches = []
        for orig_key in match_list:
            for path in original_adjlist.get(orig_key, []):
                if len(path) <= depth:  # Skip paths shorter than current depth
                    continue
                for recon_path in recon_adjlist.get(recon_key, []):
                    if len(recon_path) <= depth:  # Skip paths shorter than current depth
                        continue
                    if len(path) == len(recon_path):  # Match only paths of equal length
                        orig_adjacent_atom = orig_symbols[path[depth]]
                        recon_adjacent_atom = recon_symbols[recon_path[depth]]
                        if orig_adjacent_atom == recon_adjacent_atom:
                            if include_non_heavy:
                                # Verify non-heavy atoms connected to adjacent atoms
                                orig_non_heavy = [
                                    orig_symbols[neighbor]
                                    for neighbor in flatten(original_adjlist.get(path[depth], []))
                                    if orig_symbols[neighbor] == 'H'
                                ]

                                recon_non_heavy = [
                                    recon_symbols[neighbor]
                                    for neighbor in flatten(recon_adjlist.get(recon_path[depth], []))
                                    if recon_symbols[neighbor] == 'H'
                                ]
                                if set(orig_non_heavy) == set(recon_non_heavy):
                                    adjacent_matches.append(orig_key)
                                    break
                            else:
                                adjacent_matches.append(orig_key)
                                break
        return adjacent_matches

    # Step 1: Find initial matches based on direct symbol equality
    recon_atom = recon_symbols[recon_key]
    initial_matches = [
        orig_key for orig_key in matched_original_keys
        if orig_symbols[orig_key] == recon_atom
    ]

    # If we have a unique match or no matches, return the result
    if len(initial_matches) <= 1:
        return initial_matches

    # Step 2: Iteratively refine matches by increasing depth
    depth = 0
    current_matches = initial_matches

    while len(current_matches) > 1:
        depth += 1
        current_matches = match_paths_at_depth(depth, current_matches) if len(match_paths_at_depth(depth, current_matches)) != 0 else current_matches
        if len(current_matches) == 1:
            return current_matches
        if depth == len(recon_adjlist[recon_key]):
            break

    # Step 3: Return an empty list if no unique match is found
    return current_matches

def map_recon_to_origin(recon_xyz, orig_xyz, include_non_heavy=False):
    mapping = {}
    recon_adjlist = get_custom_adjlist(split.get_adjlist_from_dmat(recon_xyz, recon_xyz['symbols']))
    orig_adjlist = get_custom_adjlist(split.get_adjlist_from_dmat(orig_xyz, orig_xyz['symbols']))
    recon_symbols = recon_xyz['symbols']
    orig_symbols = orig_xyz['symbols']
    recon_key_list = list(recon_adjlist.keys())
    orig_key_list = list(orig_adjlist.keys())

    for recon_key in recon_key_list:
        total_lists = len(recon_adjlist[recon_key])
        matching_orig_key = []
        for orig_key in orig_key_list:
            if len(orig_adjlist[orig_key]) == total_lists:
                matching_orig_key.append(orig_key)
        if len(matching_orig_key) == 1:
            mapping[recon_key] = matching_orig_key[0]
            orig_key_list.remove(matching_orig_key[0])
        elif total_lists > 1:
            # We need to see the len of each list within list
            recon_list_count = [len(llist) for llist in recon_adjlist[recon_key]]
            llist_match = []
            for orig_key in matching_orig_key:
                orig_list_count = [len(llist) for llist in orig_adjlist[orig_key]]
                # We need to remove the recon number of lists from the orig list count and it should then be empty
                for recon_count in recon_list_count:
                    if recon_count in orig_list_count:
                        orig_list_count.remove(recon_count)
                if len(orig_list_count) == 0:
                    llist_match.append(orig_key)
            if len(llist_match) > 1:
                matches = get_adjacent_matches(recon_key, recon_symbols, recon_adjlist, llist_match, orig_symbols, orig_adjlist, include_non_heavy=include_non_heavy)
                if matches and len(matches) == 1:
                    mapping[recon_key] = matches[0]
                    orig_key_list.remove(matches[0])
                elif matches and len(matches) > 1:
                    print(f"Error: Multiple matching keys for {recon_key} found: {matches}")
                    print(f"Error: Will select the first key")
                    mapping[recon_key] = matches[0]
                    orig_key_list.remove(matches[0])
            elif len(llist_match) == 1:
                mapping[recon_key] = llist_match[0]
                orig_key_list.remove(llist_match[0])

        else:
            match_list = get_adjacent_matches(recon_key, recon_symbols, recon_adjlist, matching_orig_key, orig_symbols, orig_adjlist, include_non_heavy=include_non_heavy)
            if match_list and len(match_list) == 1:
                mapping[recon_key] = match_list[0]
                orig_key_list.remove(match_list[0])
            elif match_list and len(match_list) > 1:
                print(f"Error: Multiple matching keys for {recon_key} found: {match_list}")
                print(f"Error: Will select the first key")
                mapping[recon_key] = match_list[0]
                orig_key_list.remove(match_list[0])
            else:
                print(f"Error: No matching keys for {recon_key} found")
    return mapping
