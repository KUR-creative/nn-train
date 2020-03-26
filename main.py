#map <F4> :wa<CR>:!python % <CR>
#map <F5> :wa<CR>:!python main.py<CR>
#map <F8> :wa<CR>:!pytest -vv tests<CR>
'''
Entry point
'''
import nnlab.cli

if __name__ == '__main__':
    nnlab.cli.current_ex().run_commandline()
    
'''
def main(): 
    dset = fp.go(
        "./dataset/snet285rbk.tfrecords",
        #"./dataset/snet285wk.tfrecords",
        tf.data.TFRecordDataset,
        lambda d: dataset.read("old_snet", d))
    print(dset["num_train"])
    #train.train(dset, 4, 384, 75)
    #train.train(dset, 4, 384, 5)
    train.train(dset, 4, 384, 400)
    #train.train(dset, 4, 384, 4700)

    #inference.segment(model.Unet(),gt
    '''
