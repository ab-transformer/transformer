import pickle
from pathlib import Path
import os
import numpy as np
import torch


GT_NAMES = {
    'train': 'annotation_training.pkl',
    'valid': 'annotation_validation.pkl',
    'test': 'annotation_test.pkl'
}

SET_SIZE = {
    'train': 6000,
    'valid': 2000,
    'test': 2000
}

def load_gt(subset_name: str, gt_dir: str):
    
    
    gt_file = Path(gt_dir) / GT_NAMES[subset_name]
    
    with open(gt_file, 'rb') as f:
        gt_dict = pickle.load(f, encoding='latin1')
    target_names = list(gt_dict.keys())
    target_names.pop(-2) # remove interview
    sample_names = sorted(list(gt_dict[target_names[0]].keys()))
    gt = {Path(sample_name).stem: [gt_dict[target_name][sample_name]
          for target_name in target_names]
          for sample_name in sample_names}
    return gt, target_names
'''
def get_vid(vid_dir,shared_keys):
    temp_keys = shared_keys
    #return [np.load(os.path.join(vid_dir,key+'.npy')) if np.load(os.path.join(vid_dir,key+'.npy')).shape == (459, 2048) else temp_keys.remove(key) for key in shared_keys], temp_keys
    return [key for key in shared_keys if np.load(os.path.join(vid_dir,key+'.npy')).shape == (459, 2048)]
'''    
def load_data(subset_name: str, batch_size: int, memory_pinning: bool, drop_diff_size: bool = True):
    gt, target_names = load_gt(subset_name,'../../fodor/db/Big5/gt/')
    shared_keys = sorted(gt.keys())
    vid_dir = os.path.join("db/video/" + subset_name)
    txt_dir = os.path.join("db/text/" + subset_name)
    #aud_dir = os.path.join("db/audio" + subset_name)
    if drop_diff_size:
        shared_keys  = get_vid(vid_dir,shared_keys)
        print("Dropped shared keys")
    video = np.stack([np.expand_dims(np.load(os.path.join(vid_dir,key+'.npy')),axis=0) for key in shared_keys])
    print("Loaded video " + str(video.shape))
    text = np.stack([np.load(os.path.join(txt_dir,key+'_bertemd.npy')) for key in shared_keys],axis=0)
    print("Loaded text "  + str(text.shape))
    #audio = np.asarray([])
    labels = np.array([np.array(gt[key]) for key in shared_keys])
    ds = torch.utils.data.TensorDataset([video, text], labels)
    dl = torch.utils.data.DataLoader(ds,batch_size=batch_size,shuffle=True,pin_memory=memory_pinning)
    return dl
dl = load_data('test',10,False)