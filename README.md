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
`python -m nnlab.tasks.dataset_checking with DSET_PATH=/path/to/dset`
`python -m nnlab.tasks.train_snet with DSET_PATH=dataset/easy.tfrecord`
`python -m nnlab.tasks.train_snet with pre_learn_test DSET_PATH=dataset/easy.tfrecord`


# Omniboard
run: `omniboard -m HOST:PORT:DB` \
open: access `localhost:port` with web browser.

-m means mongo? i dunno..
