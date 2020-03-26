from nnlab.config import Ex
import nnlab.tasks.convert_old_dset

@Ex.command
def convert_old_dset(): nnlab.tasks.convert_old_dset.run()

def current_ex():
    return Ex
