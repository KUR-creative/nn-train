# nn-lab

# How to run scripts
Script files in `tasks` modules are entry points of experiments(or just task). \
run: `python -m nnlab.tasks.task_name [Sacred options]`

example \
`python -m nnlab.tasks.train_snet -m HOST:PORT:DB`

arguments of `@ex.automain`ed function are cmd args. \
example
```python
@ex.automain
def look_and_feel(dset_path):
    print(dset_path)
```
`python -m nnlab.tasks.dataset_checking with dset_path=/path/to/dset`


# Omniboard
run: `omniboard -m HOST:PORT:DB` \
open: access `localhost:port` with web browser.

-m means mongo? i dunno..
