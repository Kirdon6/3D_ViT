from collections import defaultdict
class aa_table():
    # mising some aminoacids
    # negative values????
    _hydrophobic = defaultdict(lambda:0,{
        'ALA':1,
        'ARG':0,
        'ASN':0,
        'ASP':0,
        'CYS':1,
        'GLU':0,
        'GLN':0,
        'GLY':1,
        'ILE':1,
        'LEU':1,
        'LYS':0,
        'MET':1,
        'PHE':1,
        'PRO':0, 
        'TRP':1,
        'TYR':1,
        'VAL':1,
    })
    
    _hydrophilic = defaultdict(lambda:0,{
        'ALA':0,
        'ARG':1,
        'ASN':1,
        'ASP':1,
        'CYS':0,
        'GLU':1,
        'GLN':1,
        'GLY':0,
        'ILE':0,
        'LEU':0,
        'LYS':1,
        'MET':0,
        'PHE':0,
        'PRO':1, 
        'TRP':0,
        'TYR':0,
        'VAL':0,
    })
    
    _hydrophaty_index = defaultdict(lambda:0.0,{
        'ALA':1.8,
        'ARG':-4.5,
        'ASN':-3.5,
        'ASP':-3.5,
        'CYS':2.5,
        'GLU':-3.5,
        'GLN':-3.5,
        'GLY':-0.4,
        'HIS':-3.2,
        'ILE':4.5,
        'LEU':3.8,
        'LYS':-3.9,
        'MET':1.9,
        'PHE':2.8,
        'PRO':-1.6,
        'SER':-0.8,
        'THR':-0.7,
        'TRP':-0.9,
        'TYR':-1.3,
        'VAL':4.2
    })
    
    _aliphatic = defaultdict(lambda:0,{
        'ALA':1,
        'LEU':1,
        'ILE':1,
        'VAL':1,
        'GLY':1,
        'PRO':1,
    })
    
    _aromatic = defaultdict(lambda:0,{
        'PHE':1,
        'TRP':1,
        'TYR':1,
    })
    
    _sulphur = defaultdict(lambda:0,{
        'CYS':1,
        'MET':1,
    })
    
    _hydroxyl = defaultdict(lambda:0,{
        'SER':1,
        'THR':1,
    })
    
    _basic = defaultdict(lambda:0,{
        'ARG':3,
        'LYS':2,
        'HIS':1
    })
    
    _acidic = defaultdict(lambda:0,{
        'ASP':1,
        'GLU':1
    })
    
    _amide = defaultdict(lambda:0,{
        'ASN':1,
        'GLN':1
    })
    
    _charge = defaultdict(lambda:0,{
        'ASP':-1,
        'GLU':-1,

        'ARG':1,
        'HIS':1,
        'LYS':1
    })
    
    _polar = defaultdict(lambda:0,{
        'ARG':1,
        'ASN':1,
        'ASP':1,
        'GLN':1,
        'GLU':1,
        'HIS':1,
        'LYS':1,
        'SER':1,
        'THR':1,
        'TYR':1,
        'CYS':1,
    })
    
    _ion = defaultdict(lambda:0,{
        'ASP':1,
        'GLU':1,
        'HIS':1,
        'LYS':1,
        'ARG':1,
        'CYS':1,
        'TYR':1, 
    })
    
    _donor = defaultdict(lambda:0,{
        'ARG':1,
        'LYS':1,
        'TRY':1,
    })
    
    _acceptor = defaultdict(lambda:0,{
        'ASP':1,
        'GLU':1,
    })
    
    _donoracceptor = defaultdict(lambda:0,{
        'ASN':1,
        'GLN':1,
        'HIS':1,
        'SER':1, 
        'THR':1,
        'TYR':1,
    })
    
    _propensities = defaultdict(lambda:0.0,{
        'ALA':0.701,
        'CYS':1.650,
        'ASP':1.015,
        'GLU':0.956,
        'PHE':1.952,
        'GLY':0.788,
        'HIS':2.286,
        'ILE':1.006,
        'LYS':0.468,
        'LEU':1.045,
        'MET':1.894,
        'ASN':0.811,
        'PRO':0.212,
        'GLN':0.669,
        'ARG':0.916,
        'SER':0.883,
        'THR':0.730,
        'VAL':0.884,
        'TRP':3.084,
        'TYR':1.672,
    })
     
    def get_hydrophoby(self,aa):
        return self._hydrophobic[aa]
        
    def get_hydrophily(self,aa):
        return self._hydrophilic[aa]
            
    def get_hydrophaty_index(self,aa):
        return self._hydrophaty_index[aa]
    
    def get_aliphaty(self,aa):
        return self._aliphatic[aa]
        
    def get_aromatic(self,aa):
        return self._aromatic[aa]
        
    def get_sulphuric(self,aa):
        return self._sulphur[aa]

    def get_hydroxyl(self,aa):
        return self._hydroxyl[aa]
    
    def get_basic(self,aa):
        return self._basic[aa]
        
    def get_acidic(self,aa):
        return self._acidic[aa] 
        
    def get_amide(self,aa):
        return self._amide[aa]

        
    def get_charge(self,aa):
        return self._charge[aa]
        
        
    def get_polar(self,aa):
        return self._polar[aa]

        
        
    def get_ion(self,aa):
        return self._ion[aa]
        
        
    def get_hbdonor(self,aa):
        return self._donor[aa]
        
    def get_hbacceptor(self,aa):
        return self._acceptor[aa]

        
    def get_hbdonoracceptor(self,aa):
        return self._donoracceptor[aa]

        
        
    def get_propensities(self,aa):
        return self._propensities[aa]

