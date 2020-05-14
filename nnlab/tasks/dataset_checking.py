'''
This is not an Experiment. So don't observe this script.
'''
from pathlib import Path
from sacred import Experiment
ex = Experiment(Path(__file__).stem)

'''
@ex.config
def common():
    #dset_path = None
    pass
'''

@ex.automain
def look_and_feel(dset_path):
    print(dset_path)
