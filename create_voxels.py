import Bio.PDB
import numpy as np
import math

import amino_acid_table
import at_table


pdb_file = '6zsl.pdb'
point_cloud_path = 'test.ply'

parser = Bio.PDB.PDBParser()
structure = parser.get_structure("protein", pdb_file)

atoms = structure.get_atoms()
residues = structure.get_residues()

atomic_table = at_table.at_table()
aminoacid_table = amino_acid_table.aa_table()


neighbor_search = Bio.PDB.NeighborSearch(Bio.PDB.Selection.unfold_entities(structure, 'A'))