class at_table():
    
    def __init__(self):
        self._propensities_valid, self._propensities_invalid , \
        self._propensities_sasa_valid, self._propensities_sasa_invalid, \
        self._hydrophobicity  = self.get_table_propensities()
        
        self._aromatic , self._cation , self._anion, self._hydrophobic, \
        self._acceptor , self._donor = self.get_table_volsite()
        
    
    def get_table_propensities(self):
        propensities_valid = dict()
        propensities_invalid = dict()
        propensities_sasa_valid = dict()
        propensities_sasa_invalid = dict()
        hydrophobicity = dict()
            
        with open(r'p2rank\src\main\resources\tables\atomic-properties.csv') as file:
            file.readline()
            for line in file.readlines():
                split = line.split(',')
                atomName = split[0]
            
                propensities_valid[atomName] = float(split[1])
                propensities_invalid[atomName] = float(split[2])
                propensities_sasa_valid[atomName] = float(split[3])
                propensities_sasa_invalid[atomName] = float(split[4])
                hydrophobicity[atomName] = float(split[5])
                
        return propensities_valid, propensities_invalid, propensities_sasa_valid,\
            propensities_sasa_invalid, hydrophobicity
            
    def get_table_volsite(self):
        aromatic = dict()
        cation = dict()
        anion = dict()
        hydrophobic = dict()
        acceptor = dict()
        donor = dict()
        
        with open(r'p2rank\src\main\resources\tables\volsite-atomic-properties.csv') as file:
            file.readline()
            for line in file.readlines():
                split = line.split(',')
                atomName = split[0]
            
                aromatic[atomName] = int(split[1])
                cation[atomName] = int(split[2])
                anion[atomName] = int(split[3])
                hydrophobic[atomName] = int(split[4])
                acceptor[atomName] = int(split[5])
                donor[atomName] = int(split[6])
                
        return aromatic, cation, anion,\
            hydrophobic, acceptor, donor
                
    def get_propensities_valid(self, resname, atom):
        if atom == 'OXT':
            atom = 'O'
        name = f'{resname}.{atom}'
        return self._propensities_valid[name]
    
    def get_propensities_invalid(self, resname, atom):
        if atom == 'OXT':
            atom = 'O'
        name = f'{resname}.{atom}'
        return self._propensities_invalid[name]
    
    def get_propensities_sasa_valid(self, resname, atom):
        if atom == 'OXT':
            atom = 'O'
        name = f'{resname}.{atom}'
        return self._propensities_sasa_valid[name]
    
    def get_propensities_sasa_invalid(self, resname, atom):
        if atom == 'OXT':
            atom = 'O'
        name = f'{resname}.{atom}'
        return self._propensities_sasa_invalid[name]
    
    def get_hydrophobicity(self, resname, atom):
        if atom == 'OXT':
            atom = 'O'
        name = f'{resname}.{atom}'
        return self._hydrophobicity[name]
    
    def get_volsite_aromatic(self, resname, atom):
        if atom == 'OXT':
            atom = 'O'
        name = f'{resname}.{atom}'
        return self._aromatic[name]
    
    def get_volsite_cation(self, resname, atom):
        if atom == 'OXT':
            atom = 'O'
        name = f'{resname}.{atom}'
        return self._cation[name]
    
    def get_volsite_anion(self, resname, atom):
        if atom == 'OXT':
            atom = 'O'
        name = f'{resname}.{atom}'
        return self._anion[name]
    
    def get_volsite_hydrophobic(self, resname, atom):
        if atom == 'OXT':
            atom = 'O'
        name = f'{resname}.{atom}'
        return self._hydrophobic[name]
    
    def get_volsite_acceptor(self, resname, atom):
        if atom == 'OXT':
            atom = 'O'
        name = f'{resname}.{atom}'
        return self._acceptor[name]
    
    def get_volsite_donor(self, resname, atom):
        if atom == 'OXT':
            atom = 'O'        
        name = f'{resname}.{atom}'
        return self._donor[name]
