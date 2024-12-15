import numpy as np
from arc.species.converter import str_to_xyz, xyz_to_dmat
import pandas as pd

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
    # Ensure there are at least 3 atoms in the TS
    if len(dataframe) < 3:
        raise ValueError("TS must contain at least 3 atoms.")
    if len(dataframe) == 3 and dataframe.index.str.startswith("H").sum() == 2:
        # Identify the heavy atom
        heavy_atom = dataframe.index[~dataframe.index.str.startswith("H")][0]  # Should be the only heavy atom
        hydrogen_atoms = dataframe.index[dataframe.index.str.startswith("H")]  # List of hydrogen atoms

        # Get distances from the heavy atom to the two hydrogens
        distances_to_hydrogens = dataframe.loc[heavy_atom, hydrogen_atoms]

        # Select the hydrogen with the smallest distance to the heavy atom as `H`
        hydrogen_with_min_distance = distances_to_hydrogens.idxmin()

        # The other hydrogen becomes `B`
        other_hydrogen = hydrogen_atoms[hydrogen_atoms != hydrogen_with_min_distance][0]

        return {"H": hydrogen_with_min_distance, "A": heavy_atom, "B": other_hydrogen}

    elif len(dataframe) == 4 and dataframe.index.str.startswith("H").sum() == 3:
        # Identify the heavy atom
        heavy_atom = dataframe.index[~dataframe.index.str.startswith("H")][0]  # Should be the only heavy atom
        hydrogen_atoms = dataframe.index[dataframe.index.str.startswith("H")]  # List of hydrogen atoms

        # Remove hydrogens from columns and the heavy atom from rows
        filtered_df = dataframe.loc[hydrogen_atoms, [heavy_atom]]

        # Sort the distances from the heavy atom to all hydrogens
        sorted_distances = filtered_df[heavy_atom].sort_values()

        # Select the hydrogen with the second furthest distance
        hydrogen_with_max_distance = sorted_distances.index[-2]

        # Reset the DataFrame back to the original to find the other hydrogen (`B`)
        remaining_hydrogens = hydrogen_atoms[hydrogen_atoms != hydrogen_with_max_distance]
        filtered_hydrogens_df = dataframe.loc[[hydrogen_with_max_distance], remaining_hydrogens]

        # Find the hydrogen closest to the selected hydrogen (`H`)
        closest_hydrogen = filtered_hydrogens_df.idxmin(axis=1).iloc[0]

        return {"H": hydrogen_with_max_distance, "A": heavy_atom, "B": closest_hydrogen} 

    else:

        # Filter the DataFrame for hydrogen rows and non-hydrogen columns
        hydrogen_rows = dataframe.index[dataframe.index.str.startswith("H")]
        heavy_atom_columns = dataframe.columns[~dataframe.columns.str.startswith("H")]

        filtered_df = dataframe.loc[hydrogen_rows, heavy_atom_columns]

        # Find the hydrogen atom with the smallest bond distance to a heavy atom
        min_distances = filtered_df.min(axis=1)
        min_distances = min_distances[min_distances <= 2.0]
        hydrogen_with_min_distance = min_distances.idxmax()
        min_distance_column = filtered_df.loc[hydrogen_with_min_distance].idxmin()

        # Handle cases with multiple heavy atoms
        remaining_columns = dataframe.columns[
            ~dataframe.columns.isin([hydrogen_with_min_distance, min_distance_column])
        ]
        remaining_df = dataframe.loc[[hydrogen_with_min_distance], remaining_columns]
        second_closest_atom = remaining_df.idxmin(axis=1).iloc[0]

    return {"H": hydrogen_with_min_distance, "A": min_distance_column, "B": second_closest_atom}

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
