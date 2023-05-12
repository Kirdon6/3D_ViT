import Bio.PDB
import tensorflow as tf
import numpy as np
import amino_acid_table
import at_table
import time
import os
import pickle

from positional_encodings.tf_encodings import TFPositionalEncoding3D

FEATURES_SIZE = 2
DISTANCE = 6 # (A)
PROTRUSION_DISTANCE = 10
atomic_table = at_table.at_table()
aminoacid_table = amino_acid_table.aa_table()

def calculate_distance(atom1, atom2):
    return np.linalg.norm(atom1 - atom2)


def create_dataset(protein_path='holo4k', targets_path='analyze_residues_holo4k', protein_list_path='holo4k.ds'):
    protein_list_reader = open(protein_list_path)
    start = True
    for line in protein_list_reader.readlines():
        line = line.split('/')[1].strip()
        protein = os.path.join(protein_path,line)
        target = open(os.path.join(targets_path,f"{line}_residues.csv"))
        if start:
            start = False
            target_indicator = target.readline()
        
        
        parser = Bio.PDB.PDBParser()
        structure = parser.get_structure(line, protein)
        residues = structure.get_residues()
        neighbor_search = Bio.PDB.NeighborSearch(Bio.PDB.Selection.unfold_entities(structure, 'A'))
        data = list()
        for residue in residues:
            try:
                bad_line = target.readline().split(',')
                target_indicator = int(bad_line[-1])
            except:
                print()
                
            residue_time = time.time()
            if residue.get_full_id()[3][0] == " ":

                resname = residue.get_resname()
                for atom in residue.get_atoms():
                    
                    #calculated features
                    weighted_neighbors = 0
                    carbon_count = 0
                    oxygen_count = 0
                    nitrogen_count = 0
                    donor_count = 0
                    acceptor_count = 0
                    #append aa features for atom
                    atom_name = atom.get_id()
                    #aa_time = time.time()
                    features = []
                    features.append(atom_name)
                    
                    features.append(tf.convert_to_tensor(atom.get_coord()))
                    features.append(aminoacid_table.get_hydrophoby(resname))
                    features.append(aminoacid_table.get_hydrophily(resname))
                    features.append(aminoacid_table.get_hydrophaty_index(resname))
                    features.append(aminoacid_table.get_aliphaty(resname))
                    features.append(aminoacid_table.get_aromatic(resname))
                    features.append(aminoacid_table.get_sulphuric(resname))
                    features.append(aminoacid_table.get_hydroxyl(resname))
                    features.append(aminoacid_table.get_basic(resname))
                    features.append(aminoacid_table.get_acidic(resname))
                    features.append(aminoacid_table.get_amide(resname))
                    features.append(aminoacid_table.get_charge(resname))
                    features.append(aminoacid_table.get_polar(resname))
                    features.append(aminoacid_table.get_ion(resname))
                    features.append(aminoacid_table.get_hbdonor(resname))
                    features.append(aminoacid_table.get_hbacceptor(resname))
                    features.append(aminoacid_table.get_hbdonoracceptor(resname))
                    features.append(aminoacid_table.get_propensities(resname))
                    #aa_end = time.time()
                    #print(f"AA features time:{aa_end - aa_time}")
                    #atomic_time = time.time()
                    # append atomic features for atom
                    features.append(atomic_table.get_propensities_valid(resname, atom_name))
                    features.append(atomic_table.get_propensities_invalid(resname, atom_name))
                    features.append(atomic_table.get_propensities_sasa_valid(resname, atom_name))
                    features.append(atomic_table.get_propensities_sasa_invalid(resname, atom_name))
                    features.append(atomic_table.get_hydrophobicity(resname, atom_name))
                    
                    features.append(atomic_table.get_volsite_aromatic(resname,atom_name))
                    features.append(atomic_table.get_volsite_cation(resname,atom_name))
                    features.append(atomic_table.get_volsite_anion(resname,atom_name))
                    features.append(atomic_table.get_volsite_hydrophobic(resname,atom_name))
                    features.append(atomic_table.get_volsite_acceptor(resname,atom_name))
                    features.append(atomic_table.get_volsite_donor(resname,atom_name))
                    #atomic_end = time.time()
                    #print(f"AT features time:{atomic_end - atomic_time}")
                    # calculate features
                    #calculated_time = time.time()
                    small_neighborhood = neighbor_search.search(atom.get_coord(),DISTANCE)
                    valid_small_neighborhood = list()
                    for atoms_neighborhood in small_neighborhood:
                        if atoms_neighborhood.get_full_id()[3][0] == " ":
                            valid_small_neighborhood.append(atoms_neighborhood)               
                    big_neighborhood = neighbor_search.search(atom.get_coord(), PROTRUSION_DISTANCE)
                    valid_big_neighborhood = list()
                    for atoms_neighborhood in big_neighborhood:
                        if atoms_neighborhood.get_full_id()[3][0] == " ":
                            valid_big_neighborhood.append(atoms_neighborhood)
                            
                    for neighbor_atom in valid_small_neighborhood:
                        neighbor_atom_name = neighbor_atom.get_id()
                        distance = calculate_distance(atom.get_coord(), neighbor_atom.get_coord())
                        if distance == 0:
                            continue
                        else:
                            weighted_neighbors += 1/distance
                        if neighbor_atom.get_id().startswith('C'):
                            carbon_count +=1
                        elif neighbor_atom.get_id().startswith('O'):
                            oxygen_count += 1
                        elif neighbor_atom.get_id().startswith('N'):
                            nitrogen_count += 1
                        # donor acceptor
                        if atomic_table.get_volsite_donor(neighbor_atom.get_parent().get_resname(), neighbor_atom_name) != 0:
                            donor_count += 1
                        if atomic_table.get_volsite_acceptor(neighbor_atom.get_parent().get_resname(), neighbor_atom_name) != 0:
                            acceptor_count += 1
                            
                            
                    features.append(len(valid_small_neighborhood) - 1)
                    features.append(weighted_neighbors)
                    features.append(carbon_count)
                    features.append(oxygen_count)
                    features.append(nitrogen_count)
                    features.append(donor_count)
                    features.append(acceptor_count)
                    features.append(len(valid_big_neighborhood) - 1)
                    #calculated_end = time.time()
                    #print(f"Calculated features time:{calculated_end - calculated_time}")
                    
                    #save = time.time()
                    
                    data.append((features,target_indicator))
                    '''
                    if atom.get_full_id()[2] in chains:
                        value = chains[atom.get_full_id()[2]]
                        value.append(tuple(features))               
                        chains[atom.get_full_id()[2]] = value
                    else:
                        value = [tuple(features)]
                        chains[atom.get_full_id()[2]] = value
                    '''
                    save_end = time.time()
                    
    return data
        
        
create_dataset()   

parser = Bio.PDB.PDBParser()
structure = parser.get_structure("6zsl", './2hbs.pdb')
resolution = structure.header["resolution"]
chains = dict()
atoms = structure.get_atoms()
residues = structure.get_residues()

    

count =  0



end = time.time()


        
print(count)


