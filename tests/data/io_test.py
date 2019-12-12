import yaml
from nnlab.data import io

def test_old_snet_dset():
    for yml_path in ['./dataset/snet285/indices/wk/190421wk50.yml',
                     './dataset/snet285/indices/wk/190421wk100.yml',
                     './dataset/snet285/indices/wk/190421wk150.yml', 
                     './dataset/snet285/indices/wk/190421wk200.yml', 
                     './dataset/snet285/indices/rbk/190421rbk50.yml',
                     './dataset/snet285/indices/rbk/190421rbk100.yml',
                     './dataset/snet285/indices/rbk/190421rbk150.yml',
                     './dataset/snet285/indices/rbk/190421rbk200.yml']:
        with open(yml_path) as f:
            dic = yaml.safe_load(f)
            io.old_snet_data(dic)
