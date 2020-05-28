from pathlib import Path

import tensorflow as tf
from sacred import Experiment
ex = Experiment(Path(__file__).stem)

from ..data import snet_tfrecord
from ..expr import train


@ex.named_config
def pre_learn_test():
    NUM_EPOCHS = 10
    
@ex.config
def cfg():
    NUM_EPOCHS = 1000
    BATCH_SIZE = 4
    IMG_SIZE = 384
    
@ex.automain
def train(DSET_PATH, NUM_EPOCHS, BATCH_SIZE, IMG_SIZE):
    print(DSET_PATH, NUM_EPOCHS, BATCH_SIZE, IMG_SIZE)
    dset = snet_tfrecord.read(
        tf.data.TFRecordDataset(DSET_PATH))
    train.train(dset, BATCH_SIZE, IMG_SIZE, NUM_EPOCHS)

# 지금은 정신없이 돌려봐야하는 시점이다.
# 아직 뭔가 고정시키고 목표를 만들기보다,
# 될 때까지 해야한다.
#
# 먼저 snet을 여러 파라미터로 돌려보고 (여러 모델을 쓰고)
# 이 때 Sacred로 어떻게 하는지 확인
# Sacred에 어떻게 학습 정보, 이미지를 저장하는지 확인한다.
# 테스트 결과(이미지)를 DB에 저장하는 방법.
# 테스트 결과 이미지를 실험별로 놓고 비교 가능?
# Sacred를 어떻게 쓰면 좋은지 확인한다.
