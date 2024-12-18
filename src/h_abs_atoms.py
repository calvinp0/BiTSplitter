import re

import numpy as np
import pandas as pd
from arc.species.converter import xyz_to_dmat


def extract_digits(s: str) -> int:
    """
    Extract the first integer from a string

    Args:
        s (str): The string to extract the integer from

    Returns:
        int: The first integer in the string

    """
    return int(re.sub(r"[^\d]", "", s))


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

    closest_atoms = {}
    for index, row in dataframe.iterrows():

        row[index] = np.inf
        closest = row.nsmallest(2).index.tolist()
        closest_atoms[index] = closest

    hydrogen_keys = [key for key in dataframe.index if key.startswith("H")]
    condition_occurrences = []

    for hydrogen_key in hydrogen_keys:
        atom_neighbours = closest_atoms[hydrogen_key]
        is_heavy_present = any(
            atom for atom in closest_atoms if not atom.startswith("H")
        )
        if_hydrogen_present = any(
            atom
            for atom in closest_atoms
            if atom.startswith("H") and atom != hydrogen_key
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
            return {
                "H": extract_digits(best_occurrence["H"]),
                "A": extract_digits(best_occurrence["A"]),
                "B": extract_digits(best_occurrence["B"]),
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
            return {
                "H": extract_digits(selected_hydrogen),
                "A": extract_digits(selected_heavy_atoms[0]),
                "B": extract_digits(selected_heavy_atoms[1]),
            }
        else:
            raise ValueError("No valid hydrogen atom found.")


def find_valid_atoms(hydrogen_with_min_distance, min_distances, filtered_df, dataframe):
    """
    Recursively find a valid hydrogen atom and its second closest atom that meets the distance criteria.

    Args:
        hydrogen_with_min_distance (str): The current hydrogen atom under consideration.
        min_distances (pd.Series): Series containing minimum distances of hydrogen atoms to heavy atoms.
        filtered_df (pd.DataFrame): Filtered dataframe of hydrogen rows and heavy atom columns.
        dataframe (pd.DataFrame): The full distance dataframe.

    Returns:
        tuple: Valid hydrogen atom and its second closest atom.
    """
    min_distance_column = filtered_df.loc[hydrogen_with_min_distance].idxmin()
    remaining_columns = dataframe.columns[
        ~dataframe.columns.isin([hydrogen_with_min_distance, min_distance_column])
    ]
    remaining_df = dataframe.loc[[hydrogen_with_min_distance], remaining_columns]
    second_closest_atom = remaining_df.idxmin(axis=1).iloc[0]

    if "H" in second_closest_atom:
        # Check their distances to the hydrogen atom
        distances_to_second_closest_atom = dataframe.loc[
            hydrogen_with_min_distance, second_closest_atom
        ]
        if distances_to_second_closest_atom > 1.7:
            # Remove the current hydrogen atom and retry
            min_distances = min_distances.drop(hydrogen_with_min_distance)

            # Terminate if no more hydrogen atoms remain
            if min_distances.empty:
                raise ValueError("No valid hydrogen atom found.")

            # Retry with the next hydrogen atom
            hydrogen_with_min_distance = min_distances.idxmin()
            return find_valid_atoms(
                hydrogen_with_min_distance, min_distances, filtered_df, dataframe
            )

    return hydrogen_with_min_distance, min_distance_column, second_closest_atom


def pull_atoms_closer(xyz, h_index, a_index, target_distance=1.0):
    """
    Adjusts the position of atom at h_index to be closer to atom at a_index by setting the distance between them to target_distance.

    Parameters:
    - xyz (dict): A dictionary containing atomic coordinates with key 'coords'.
                  Example: {'coords': np.array([[x1, y1, z1], [x2, y2, z2], ...])}
    - h_index (int): Index of the atom to be moved closer.
    - a_index (int): Index of the reference atom.
    - target_distance (float): Desired distance between the two atoms in angstroms (default is 1.0).

    Returns:
    - dict: Updated xyz dictionary with modified coordinates.
    """
    # Extract coordinates of the two atoms
    h_coords = np.array(xyz["coords"][h_index])
    a_coords = np.array(xyz["coords"][a_index])

    # Calculate the vector from a_index to h_index
    vector = h_coords - a_coords
    distance = np.linalg.norm(vector)

    if distance == 0:
        raise ValueError(
            "The two atoms are at the same position; direction is undefined."
        )

    # Normalize the vector and scale it to the target distance
    unit_vector = vector / distance
    new_vector = unit_vector * target_distance

    # Calculate the new coordinates for h_index
    new_h_coords = a_coords + new_vector

    # Update the coordinates in the xyz dictionary
    updated_xyz = xyz.copy()
    updated_xyz["coords"] = [
        list(coord) for coord in xyz["coords"]
    ]  # Ensure it's mutable
    updated_xyz["coords"][
        h_index
    ] = new_h_coords.tolist()  # Convert back to list if necessary

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
