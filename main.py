#map <F5> :wa<CR>:!python main.py<CR>

from bidict import bidict

from nnlab.tasks import dataset
from nnlab.utils import file_utils as fu


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
    #./dataset/snet285/indices/rbk/190421rbk200.yml

if __name__ == '__main__':
    main()
