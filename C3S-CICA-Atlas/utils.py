import yaml

class YAMLconfig(dict):
    def __init__(self, data):
        if isinstance(data, str):
            with open(data, 'r') as file:
                data = yaml.safe_load(file)
        super().__init__(data)
        
    def __getattr__(self, name):
        if name in self:
            value = self[name]
            if isinstance(value, dict):
                return YAMLconfig(value)
            return value
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        self[name] = value
        
    def __delattr__(self, name):
        del self[name]


RCM_DICT = {
    'EUR-11': 
    { 
        'REMO': 'GERICS_REMO2015',
        'RegCM': 'ICTP_RegCM4-6',
    },
    'EUR-22': 
    {
        'REMO': 'GERICS_REMO2015',
    },
    'WAS-22': {
        'REMO': 'GERICS_REMO2015',
        'RegCM': 'ICTP_RegCM4-7',
    },
    'EAS-22':
     {
        'REMO': 'GERICS_REMO2015',
        'RegCM': 'KNU_RegCM4-0',
    },
    'CAM-22':
     {
        'REMO': 'GERICS_REMO2015',
        'RegCM': 'ICTP_RegCM4-7',
    },
    'SAM-22':
     {
        'REMO': 'GERICS_REMO2015',
        'RegCM': 'ICTP_RegCM4-7',
    },
    'NAM-22':
     {
        'REMO': 'GERICS_REMO2015',
        'RegCM': 'ICTP_RegCM4-6',
    },
    'AUS-22':
     {
        'REMO': 'GERICS_REMO2015',
        'RegCM': 'ICTP_RegCM4-7',
    },
    'AFR-22':
     {
        'REMO': 'GERICS_REMO2015',
        'RegCM': 'ICTP_RegCM4-7',
    },
    'SEA-22':
    {
        'REMO': 'GERICS_REMO2015',
        'RegCM': 'ICTP_RegCM4-7',
    },
}


MODEL_DICT={
    'REMO' : dict(sftuf='orig_v3', orog='orog',sftlf='sftlf'),
    'RegCM' : dict(sftuf='', orog='orog',sftlf='sftlf'),
}
