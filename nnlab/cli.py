from nnlab.config import Ex

import nnlab.tasks.convert_old_dset
import nnlab.tasks.train

#--------------------------------------------------------------
@Ex.command
def convert_old_dset(): nnlab.tasks.convert_old_dset.run()

@Ex.command
def train(): nnlab.tasks.train.run()

#--------------------------------------------------------------
def current_ex():
    return Ex
