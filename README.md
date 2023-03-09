# nn-lab
## Summary
![szmc architecture](https://github.com/KUR-creative/SickZil-Machine/raw/master/doc/szmc-structure-eng.png)
딥러닝 모델 학습을 위한 레포입니다. 이를 통해 만화 이미지에서 텍스트 영역을 추론하는 SegNet을 학습시켰습니다.
- 기존 [U-net](https://arxiv.org/abs/1505.04597)에 Batch Normalization을 적용하고 deconvolution을 적용하는 등 성능을 위해 모델을 수정하였습니다.
- [여기](https://github.com/KUR-creative/old-nn-lab/blob/master/nnlab/nn/model.py)서 모델 코드를 확인할 수 있습니다.
- 모델 코드에는 tf2를, 데이터셋으로는 tfrecord를 적용하였습니다.

## How to run scripts
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
