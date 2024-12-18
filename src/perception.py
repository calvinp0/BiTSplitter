from arc.species import ARCSpecies


def check_isomorphism(mol1_xyz: dict, mol2_xyz: dict) -> bool:
    """
    Check if two molecules are the same.
    """
    mol1 = ARCSpecies(label="mol1", xyz=mol1_xyz)
    mol2 = ARCSpecies(label="mol2", xyz=mol2_xyz)

    isomorphic = mol1.is_isomorphic(mol2)
    if not isomorphic:
        return False
    return True
