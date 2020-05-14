from pathlib import Path
from sacred import Experiment
ex = Experiment(Path(__file__).stem)

@ex.automain
def ex1():
    print('ppap')
