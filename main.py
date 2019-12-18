#map <F5> :wa<CR>:!python main.py<CR>

from bidict import bidict

from nnlab.tasks import dataset
from nnlab.utils import file_utils as fu

import yaml

'''
`img_dirpath` and `mask_dirpath` are leaf directory 
that contains image in flat manner. (Only contains images, no recursive 
directory structure.)
'''

'''
# Get image,mask sequence.
img_paths  = fu.human_sorted(fu.children('./dataset/snet285/image'))
mask_paths = fu.human_sorted(fu.children('./dataset/snet285/clean_rbk'))
num_imgs = len(img_paths)
num_masks = len(mask_paths)
assert num_imgs == num_masks

dataset.generate(
    img_paths, mask_paths,
    bidict({(255,  0,  0): (0,0,1),
            (  0,  0,255): (0,1,0),
            (  0,  0,  0): (1,0,0)}),
    './dataset/snet285rbk.tfrecords')

# Get image,mask sequence.
img_paths  = fu.human_sorted(fu.children('./dataset/snet285/image'))
mask_paths = fu.human_sorted(fu.children('./dataset/snet285/clean_wk'))
num_imgs = len(img_paths)
num_masks = len(mask_paths)
assert num_imgs == num_masks

dataset.generate(
    img_paths, mask_paths,
    bidict({(255,255,255): (0,1),
            (  0,  0,  0): (1,0)}),
    './dataset/snet285wk.tfrecords')
'''

def main():
    with open('dataset/snet285/indices/rbk/190421rbk200.yml') as f:
        dset_dic = yaml.safe_load(f)

    dset = dataset.distill('old_snet', dset_dic)
    train_path_pairs = dset['train']
    valid_path_pairs = dset['valid']
    test_path_pairs  = dset['test']
    src_dst_colormap = dset['cmap']

    print(train_path_pairs)
    print(valid_path_pairs)
    print(test_path_pairs)
    print(src_dst_colormap)

    dataset.generate(
        train_path_pairs, valid_path_pairs, test_path_pairs,
        src_dst_colormap, 'test_dset.tfrecords', 
        look_and_feel_check=True)

if __name__ == '__main__':
    main()
