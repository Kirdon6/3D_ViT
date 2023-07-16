import Bio.PDB
import tensorflow as tf
import numpy as np
import amino_acid_table
import at_table
import os
import argparse



DISTANCE = 6 # (A)
PROTRUSION_DISTANCE = 10
ATOMIC_TABLE = at_table.at_table()
AA_TABLE = amino_acid_table.aa_table()
ATOM_TYPES = 36

parser = argparse.ArgumentParser()
parser.add_argument("--create_dataset", default=True, type=bool,help="If True creates and saves dataset otherwise just loads.")
parser.add_argument("--new_file_name", default="dataset_HOLO4k", type=str, help="Path for creating dataset.")
parser.add_argument("--input_file", default="dataset_HOLO4k.npz", type=str, help="Path to saved dataset.")

def calculate_distance(atom1, atom2):
    return np.linalg.norm(atom1 - atom2)


def create_dataset(protein_path='holo4k', targets_path='analyze_residues_holo4k', protein_list_path='holo4k.ds'):
    protein_list_reader = open(protein_list_path)
    start = True
    counter =  0

    files = list()
    for line in protein_list_reader.readlines():
        counter += 1
        print(f"Protein number: {counter}")
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
                    
                    atom_name = atom.get_id()
                    
                    features = []
                    # atom_name
                    features.append(atom_name)
                    
                    # coordinates
                    for x in atom.get_coord():
                        features.append(x)
                        
                    #categorical features
                    features.append(AA_TABLE.get_hydrophoby(resname))
                    features.append(AA_TABLE.get_hydrophily(resname))                    
                    features.append(AA_TABLE.get_aliphaty(resname))
                    features.append(AA_TABLE.get_aromatic(resname))
                    features.append(AA_TABLE.get_sulphuric(resname))
                    features.append(AA_TABLE.get_hydroxyl(resname))
                    features.append(AA_TABLE.get_basic(resname))
                    features.append(AA_TABLE.get_acidic(resname))
                    features.append(AA_TABLE.get_amide(resname))
                    features.append(AA_TABLE.get_charge(resname))
                    features.append(AA_TABLE.get_polar(resname))
                    features.append(AA_TABLE.get_ion(resname))
                    features.append(AA_TABLE.get_hbdonor(resname))
                    features.append(AA_TABLE.get_hbacceptor(resname))
                    features.append(AA_TABLE.get_hbdonoracceptor(resname))
                    features.append(ATOMIC_TABLE.get_volsite_aromatic(resname,atom_name))
                    features.append(ATOMIC_TABLE.get_volsite_cation(resname,atom_name))
                    features.append(ATOMIC_TABLE.get_volsite_anion(resname,atom_name))
                    features.append(ATOMIC_TABLE.get_volsite_hydrophobic(resname,atom_name))
                    features.append(ATOMIC_TABLE.get_volsite_acceptor(resname,atom_name))
                    features.append(ATOMIC_TABLE.get_volsite_donor(resname,atom_name))
                    
                    # numerical values
                    features.append(atom.get_bfactor())                    
                    features.append(AA_TABLE.get_hydrophaty_index(resname))
                    features.append(AA_TABLE.get_propensities(resname))                    
                    features.append(ATOMIC_TABLE.get_propensities_valid(resname, atom_name))
                    features.append(ATOMIC_TABLE.get_propensities_invalid(resname, atom_name))
                    features.append(ATOMIC_TABLE.get_propensities_sasa_valid(resname, atom_name))
                    features.append(ATOMIC_TABLE.get_propensities_sasa_invalid(resname, atom_name))
                    features.append(ATOMIC_TABLE.get_hydrophobicity(resname, atom_name))                    


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
                        if ATOMIC_TABLE.get_volsite_donor(neighbor_atom.get_parent().get_resname(), neighbor_atom_name) != 0:
                            donor_count += 1
                        if ATOMIC_TABLE.get_volsite_acceptor(neighbor_atom.get_parent().get_resname(), neighbor_atom_name) != 0:
                            acceptor_count += 1
                            
                            
                    features.append(len(valid_small_neighborhood) - 1)
                    features.append(weighted_neighbors)
                    features.append(carbon_count)
                    features.append(oxygen_count)
                    features.append(nitrogen_count)
                    features.append(donor_count)
                    features.append(acceptor_count)
                    features.append(len(valid_big_neighborhood) - 1)
                    
                    
                    data.append((features,float(target_indicator)))

        files.append(data)               
    return files

def separate_tuples(tensor):
    first, second = list(), list()
    for i in range(len(tensor)):
        first.append(tensor[i][0])
        second.append(tensor[i][1])
    return first, second
def extract_first(tensor):
    first_items = [sublist[0] for sublist in tensor]
    rest = [sublist[1:] for sublist in tensor]
    return first_items , rest

def get_unique_length(my_list):
    unique_length = len(set(my_list))
    return unique_length

def one_hot_encode_strings(string_list):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=len(set(string_list)))
    tokenizer.fit_on_texts(string_list)

    sequences = tokenizer.texts_to_sequences(string_list)
    flat = list()
    for sublist in sequences:
        if sublist:
            flat.append([sublist[0]])
        else:
            flat.append([0])


    return flat

def create_one_hot_atoms(tensor):

    # Extract the first position from every feature vector
    first_positions, rest = extract_first(tensor=tensor)
    encoded_positions = tf.convert_to_tensor(one_hot_encode_strings(first_positions), dtype=np.float32)
    rest = tf.convert_to_tensor(rest)
    concatenated = tf.concat([encoded_positions,rest],axis=1)


    return concatenated

def find_all_atom_types(protein_path='holo4k', targets_path='analyze_residues_holo4k', protein_list_path='holo4k.ds'):
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
        atoms = list()
        for residue in residues:
            try:
                bad_line = target.readline().split(',')
                target_indicator = int(bad_line[-1])
            except:
                print()
                
            #residue_time = time.time()
            if residue.get_full_id()[3][0] == " ":
                for atom in residue.get_atoms():
                    
                    #calculated features
                    #append aa features for atom
                    atoms.append(atom.get_id())
                    
    return atoms

def stack_ragged(tensors):
    values = tf.concat(tensors, axis=0)
    lens = tf.stack([tf.shape(t, out_type=tf.int64)[0] for t in tensors])
    return tf.RaggedTensor.from_row_lengths(values, lens)

def split_dataset(data):
    targets = list()
    for value in range(len(data)):
        first, second = separate_tuples(data[value])
        data[value] = tf.convert_to_tensor(create_one_hot_atoms(first))
        targets.append(tf.convert_to_tensor(second))
        
    return data, targets

def load_dataset(args):
    if args.create_dataset:
        data = create_dataset()
        data, targets = split_dataset(data)
        np.savez_compressed(args.new_file_name,data = data, targets=targets)
    holo4k = np.load(args.input_file, allow_pickle=True)
    data = holo4k["data"]
    targets = holo4k["targets"]
    return data, targets


# TODO deprecated
if __name__ == "__main__":  
    args = parser.parse_args()
    if args.create_dataset:
        data = create_dataset()
        data, targets = split_dataset(data)
        np.savez_compressed(args.new_file_name,data = data, targets=targets)
    holo4k = np.load(args.input_file, allow_pickle=True)
    data = holo4k["data"]
    targets = holo4k["targets"]
