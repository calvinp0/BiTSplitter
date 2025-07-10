import re
from typing import Tuple, Dict, List, Set


import numpy as np
import pandas as pd
from arc.species.converter import xyz_to_dmat
from arc.common import SINGLE_BOND_LENGTH

### Changing H_H length from 1.0 to 0.74
SINGLE_BOND_LENGTH["H_H"] = 0.74  # Å, this is the length of the H-H bond in H2

from split import bonded, get_adjlist_from_dmat

ATOMIC_MASSES = {'H' : 1.008,'HE' : 4.003, 'LI' : 6.941, 'BE' : 9.012,\
                 'B' : 10.811, 'C' : 12.011, 'N' : 14.007, 'O' : 15.999,\
                 'F' : 18.998, 'NE' : 20.180, 'NA' : 22.990, 'MG' : 24.305,\
                 'AL' : 26.982, 'SI' : 28.086, 'P' : 30.974, 'S' : 32.066,\
                 'CL' : 35.453, 'AR' : 39.948, 'K' : 39.098, 'CA' : 40.078,\
                 'SC' : 44.956, 'TI' : 47.867, 'V' : 50.942, 'CR' : 51.996,\
                 'MN' : 54.938, 'FE' : 55.845, 'CO' : 58.933, 'NI' : 58.693,\
                 'CU' : 63.546, 'ZN' : 65.38, 'GA' : 69.723, 'GE' : 72.631,\
                 'AS' : 74.922, 'SE' : 78.971, 'BR' : 79.904, 'KR' : 84.798,\
                 'RB' : 84.468, 'SR' : 87.62, 'Y' : 88.906, 'ZR' : 91.224,\
                 'NB' : 92.906, 'MO' : 95.95, 'TC' : 98.907, 'RU' : 101.07,\
                 'RH' : 102.906, 'PD' : 106.42, 'AG' : 107.868, 'CD' : 112.414,\
                 'IN' : 114.818, 'SN' : 118.711, 'SB' : 121.760, 'TE' : 126.7,\
                 'I' : 126.904, 'XE' : 131.294, 'CS' : 132.905, 'BA' : 137.328,\
                 'LA' : 138.905, 'CE' : 140.116, 'PR' : 140.908, 'ND' : 144.243,\
                 'PM' : 144.913, 'SM' : 150.36, 'EU' : 151.964, 'GD' : 157.25,\
                 'TB' : 158.925, 'DY': 162.500, 'HO' : 164.930, 'ER' : 167.259,\
                 'TM' : 168.934, 'YB' : 173.055, 'LU' : 174.967, 'HF' : 178.49,\
                 'TA' : 180.948, 'W' : 183.84, 'RE' : 186.207, 'OS' : 190.23,\
                 'IR' : 192.217, 'PT' : 195.085, 'AU' : 196.967, 'HG' : 200.592,\
                 'TL' : 204.383, 'PB' : 207.2, 'BI' : 208.980, 'PO' : 208.982,\
                 'AT' : 209.987, 'RN' : 222.081, 'FR' : 223.020, 'RA' : 226.025,\
                 'AC' : 227.028, 'TH' : 232.038, 'PA' : 231.036, 'U' : 238.029,\
                 'NP' : 237, 'PU' : 244, 'AM' : 243, 'CM' : 247, 'BK' : 247,\
                 'CT' : 251, 'ES' : 252, 'FM' : 257, 'MD' : 258, 'NO' : 259,\
                 'LR' : 262, 'RF' : 261, 'DB' : 262, 'SG' : 266, 'BH' : 264,\
                 'HS' : 269, 'MT' : 268, 'DS' : 271, 'RG' : 272, 'CN' : 285,\
                 'NH' : 284, 'FL' : 289, 'MC' : 288, 'LV' : 292, 'TS' : 294,\
                 'OG' : 294}


def extract_digits(s: str) -> int:
    """
    Extract the first integer from a string

    Args:
        s (str): The string to extract the integer from

    Returns:
        int: The first integer in the string

    """
    return int(re.sub(r"[^\d]", "", s))

def get_element_symbol(atom_label: str) -> str:
    """
    Get the element symbol from an atom label

    Args:
        atom_label (str): The atom label

    Returns:
        str: The element symbol
    """
    match = re.match(r"^([A-Za-z]{1,2})", atom_label)
    if not match:
        raise ValueError(f"Could not extract element symbol from {atom_label}")
    return match.group(1)


def chain_weight(atom_idx: int, adjacency: Dict[int, List[int]], symbols: Tuple[str, ...], visited: Set[int] = None) -> float:
    """
    Recursively calculate the weight of a chain of atoms
    """
    if visited is None:
        visited = set()
    if atom_idx in visited:
        return 0.0

    visited.add(atom_idx)
    # e.g. symbols[3] == 'C'
    symbol = symbols[atom_idx]
    mass_here = ATOMIC_MASSES.get(symbol, 0.0)

    total = mass_here
    for nb in adjacency.get(atom_idx, []):
        if nb not in visited and not symbols[nb].startswith('H'):
            total += chain_weight(nb, adjacency, symbols, visited)
    return total


def get_best_heavy_neighbour(atom_idx: int, adjacency: Dict[int, List[int]], symbols: Tuple[str, ...], exclude_idx: int = None) -> int:
    """
    Among all neighbors of `atom_idx` in adjacency, pick the neighbor
    whose chain weight is largest, provided that neighbor is "heavy" (i.e. not 'H').

    Args:
        atom_idx (int): The atom index.
        adjacency (dict[int, list[int]]): The adjacency dict.
        symbols (tuple[str, ...]): The element symbols.
        exclude_idx (int, optional): If given, skip this neighbor.

    Returns:
        int or None: The index of the best heavy neighbor (or None if none).
    """

    candidates = []
    hydrogen_candidates = []
    for nb_idx in adjacency.get(atom_idx, []):
        if nb_idx == exclude_idx:
            continue
        # e.g. if symbols[nb_idx] == 'H', skip
        if not symbols[nb_idx].startswith('H'):
            candidates.append(nb_idx)
        elif symbols[nb_idx].startswith('H'):
            hydrogen_candidates.append(nb_idx)

    if candidates:
        best_candidate = None
        best_weight = -1.0
        for cand_idx in candidates:
            w = chain_weight(cand_idx, adjacency, symbols, visited={atom_idx})
            if w > best_weight:
                best_weight = w
                best_candidate = cand_idx
            elif w == best_weight:
                # Pick the candidate that is the heaviest atom
                if ATOMIC_MASSES.get(symbols[cand_idx], 0.0) > ATOMIC_MASSES.get(symbols[best_candidate], 0.0):
                    best_candidate = cand_idx
        return best_candidate

    elif hydrogen_candidates:
        return hydrogen_candidates[0]
    elif not candidates and not hydrogen_candidates:
        return None

def _get_next_heavy_neighbour(adjacency: dict, A: str, H: str, B: str, symbols: list) -> str:

    C = get_best_heavy_neighbour(A, adjacency, symbols=symbols, exclude_idx=H)
    D = get_best_heavy_neighbour(B, adjacency, symbols=symbols, exclude_idx=H)

    return C, D



def convert_xyz_to_df(xyz: dict) -> pd.DataFrame:
    """
    Convert a dictionary of xyz coords to a pandas DataFrame with bond distances

    Args:
        xyz (dict): The xyz coordinates of the molecule

    Return:
        pd.DataFrame: The xyz coordinates as a pandas DataFrame

    """
    symbols = xyz["symbols"]
    symbol_enum = [f"{symbol}{i}" for i, symbol in enumerate(symbols)]
    ts_dmat = xyz_to_dmat(xyz)

    return pd.DataFrame(ts_dmat, columns=symbol_enum, index=symbol_enum)


def get_h_abs_atoms(dataframe: pd.DataFrame) -> dict:
    """
    Get the donating/accepting hydrogen atom, and the two heavy atoms that are bonded to it

    Args:
        dataframe (pd.DataFrame): The dataframe of the bond distances, columns and index are the atom symbols

    Returns:
        dict: The hydrogen atom and the two heavy atoms. The keys are 'H', 'A', 'B'
    """
    symbols = [get_element_symbol(symbol) for symbol in dataframe.columns]
    adjacency = get_adjlist_from_dmat(dataframe.values, symbols)

    # -------------------------------------------------------------------
    # 1. Build a dictionary with each atom's two closest neighbors
    # -------------------------------------------------------------------
    closest_atoms = {}
    for index, row in dataframe.iterrows():

        row[index] = np.inf
        closest = row.nsmallest(2).index.tolist()
        closest_atoms[index] = closest

    # -------------------------------------------------------------------
    # 2. Identify which atoms are hydrogen
    # -------------------------------------------------------------------
    hydrogen_keys = [key for key in dataframe.index if key.startswith("H")]
    condition_occurrences = []

    # -------------------------------------------------------------------
    # 3. Gather valid H-A-B occurrences
    # -------------------------------------------------------------------
    for hydrogen_key in hydrogen_keys:
        atom_neighbours = closest_atoms[hydrogen_key]
        is_heavy_present = any(
            atom for atom in atom_neighbours if not atom.startswith("H")
        )
        # if_hydrogen_present = any(
        #     atom
        #     for atom in atom_neighbours
        #     if atom.startswith("H") and atom != hydrogen_key
        # )
        if_hydrogen_present = (
        atom_neighbours[0].startswith("H") and atom_neighbours[0] != hydrogen_key
    )

        if is_heavy_present and if_hydrogen_present:
            # Store the details of this occurrence
            condition_occurrences.append(
                {"H": hydrogen_key, "A": atom_neighbours[0], "B": atom_neighbours[1]}
            )

    # Check if the condition was met
    if condition_occurrences:
        if len(condition_occurrences) > 1:
            # Store distances to decide which occurrence to use
            occurrence_distances = []
            for occurrence in condition_occurrences:
                # Calculate the sum of distances to the two heavy atoms
                hydrogen_key = f"{occurrence['H']}"
                heavy_atoms = [f"{occurrence['A']}", f"{occurrence['B']}"]
                try:
                    distances = dataframe.loc[hydrogen_key, heavy_atoms].sum()
                    occurrence_distances.append((occurrence, distances))
                except KeyError as e:
                    print(f"Error accessing distances for occurrence {occurrence}: {e}")

            # Select the occurrence with the smallest distance
            best_occurrence = min(occurrence_distances, key=lambda x: x[1])[0]
            C, D = _get_next_heavy_neighbour(adjacency, extract_digits(best_occurrence["A"]), extract_digits(best_occurrence["H"]), extract_digits(best_occurrence["B"]), symbols)
            return {
                "H": extract_digits(best_occurrence["H"]),
                "A": extract_digits(best_occurrence["A"]),
                "B": extract_digits(best_occurrence["B"]),
                "C": C,
                "D": D
            }
        else:
            C, D = _get_next_heavy_neighbour(adjacency, condition_occurrences[0]["A"], condition_occurrences[0]["H"], condition_occurrences[0]["B"], symbols)
            return {
                "H": extract_digits(condition_occurrences[0]["H"]),
                "A": extract_digits(condition_occurrences[0]["A"]),
                "B": extract_digits(condition_occurrences[0]["B"]),
                "C": C,
                "D": D
            }
    else:

        # Check the all the hydrogen atoms, and see the closest two heavy atoms and aggregate their distances to determine which Hyodrogen atom has the lowest distance aggregate
        min_distance = np.inf
        selected_hydrogen = None
        selected_heavy_atoms = None

        for hydrogen_key in hydrogen_keys:
            atom_neighbours = closest_atoms[hydrogen_key]
            heavy_atoms = [atom for atom in atom_neighbours if not atom.startswith("H")]

            if len(heavy_atoms) < 2:
                continue

            distances = dataframe.loc[hydrogen_key, heavy_atoms[:2]].sum()
            if distances < min_distance:
                min_distance = distances
                selected_hydrogen = hydrogen_key
                selected_heavy_atoms = heavy_atoms

        if selected_hydrogen:
            C, D = _get_next_heavy_neighbour(adjacency, extract_digits(selected_heavy_atoms[0]), extract_digits(selected_hydrogen), extract_digits(selected_heavy_atoms[1]), symbols)
            return {
                "H": extract_digits(selected_hydrogen),
                "A": extract_digits(selected_heavy_atoms[0]),
                "B": extract_digits(selected_heavy_atoms[1]),
                "C": C,
                "D": D
            }
        else:
            raise ValueError("No valid hydrogen atom found.")




def pull_atoms_closer(
    xyz, h_index, a_index, target_distance=1.0, tolerance=1.2
):
    """
    Adjust the position of the hydrogen at h_index so its distance to a_index matches:
    - The average of sibling Hs bonded to a_index, or
    - The SINGLE_BOND_LENGTH lookup for the pair, or
    - target_distance as a final fallback.

    Parameters:
    - xyz (dict): Dictionary with 'symbols', 'isotopes', and 'coords' (tuple of floats).
    - h_index (int): Index of the hydrogen atom to adjust.
    - a_index (int): Index of the heavy atom it should be bonded to.
    - target_distance (float): Fallback distance if no siblings or no bond length is found.
    - tolerance (float): Max distance (Å) to consider other hydrogens bonded to a_index (default 1.2 Å).
    - SINGLE_BOND_LENGTH (dict): Dictionary of bond lengths by element pair (e.g., 'H_O': 0.96) - Provided in the script as global variable.

    Returns:
    - dict: Updated xyz dictionary with modified coordinates.
    """
    coords = np.array(xyz["coords"])
    symbols = xyz["symbols"]
    a_coords = coords[a_index]
    a_symbol = symbols[a_index]
    h_symbol = symbols[h_index]

    # Special case: If a_index is H, skip sibling search and just use bond lookup or fallback
    if a_symbol == "H":
        key1 = f"{h_symbol}_{a_symbol}"
        key2 = f"{a_symbol}_{h_symbol}"
        if SINGLE_BOND_LENGTH:
            avg_distance = SINGLE_BOND_LENGTH.get(key1) or SINGLE_BOND_LENGTH.get(key2) or target_distance
        else:
            avg_distance = target_distance
    else:
        # Find sibling hydrogens connected to a_index (excluding h_index)
        other_h_indices = []
        for i, (coord, symbol) in enumerate(zip(coords, symbols)):
            if i == h_index:
                continue
            if symbol == "H":
                distance = np.linalg.norm(coord - a_coords)
                if distance < tolerance:
                    other_h_indices.append(i)

        if other_h_indices:
            avg_distance = np.mean([np.linalg.norm(coords[i] - a_coords) for i in other_h_indices])
        else:
            key1 = f"{h_symbol}_{a_symbol}"
            key2 = f"{a_symbol}_{h_symbol}"
            if SINGLE_BOND_LENGTH:
                avg_distance = SINGLE_BOND_LENGTH.get(key1) or SINGLE_BOND_LENGTH.get(key2) or target_distance
            else:
                avg_distance = target_distance

    # Move the hydrogen to the desired distance
    h_coords = coords[h_index]
    vector = h_coords - a_coords
    distance = np.linalg.norm(vector)

    if distance == 0:
        # Fallback arbitrary direction if atoms overlap
        vector = np.array([1.0, 0.0, 0.0])
        distance = 1.0

    unit_vector = vector / distance
    new_h_coords = a_coords + unit_vector * avg_distance

    # Update the xyz dictionary
    updated_xyz = xyz.copy()
    updated_xyz["coords"] = [list(coord) for coord in coords]
    updated_xyz["coords"][h_index] = new_h_coords.tolist()

    return updated_xyz


def batch_process_h_abs_atoms(xyz_list: list) -> list:
    """
    Process a batch of xyz dictionaries to extract hydrogen abstraction atoms.

    Args:
        xyz_list (list): A list of xyz dictionaries.

    Returns:
        list: A list of dictionaries, each containing the 'H', 'A', and 'B' atoms for a molecule.
    """
    results = []
    for i, xyz in enumerate(xyz_list):
        try:
            dataframe = convert_xyz_to_df(xyz)
            h_abs_atoms = get_h_abs_atoms(dataframe)
            h_abs_atoms["TS"] = xyz
            results.append(h_abs_atoms)
        except ValueError as e:
            print(f"Error processing molecule {i}: {e}")
            results.append({"H": None, "A": None, "B": None, "TS": xyz})
    return results
