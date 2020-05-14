# nn-lab

# How to run scripts
Script files in `tasks` modules are entry points of experiments(or just task). \
run: `python -m nnlab.tasks.task_name [Sacred options]`

example \
`python -m nnlab.tasks.train_snet -m HOST:PORT:DB`


# Omniboard
run: `omniboard -m HOST:PORT:DB` \
open: access `localhost:port` with web browser.

-m means mongo? i dunno..
