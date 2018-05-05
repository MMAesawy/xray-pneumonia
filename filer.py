import os
import re
import shutil

new_dir = './chest_xray_3class'
os.mkdir(new_dir)
os.mkdir(os.path.join(new_dir, 'train'))
os.mkdir(os.path.join(new_dir, 'test'))
os.mkdir(os.path.join(new_dir, 'val'))

new_train_normal_dir = os.path.join(new_dir, 'train', 'NORMAL')
new_train_bact_dir = os.path.join(new_dir, 'train', 'BACTERIAL')
new_train_viral_dir = os.path.join(new_dir, 'train', 'VIRAL')
new_test_normal_dir = os.path.join(new_dir, 'test', 'NORMAL')
new_test_bact_dir = os.path.join(new_dir, 'test', 'BACTERIAL')
new_test_viral_dir = os.path.join(new_dir, 'test', 'VIRAL')
new_val_normal_dir = os.path.join(new_dir, 'val', 'NORMAL')
new_val_bact_dir = os.path.join(new_dir, 'val', 'BACTERIAL')
new_val_viral_dir = os.path.join(new_dir, 'val', 'VIRAL')

os.mkdir(new_train_normal_dir)
os.mkdir(new_train_bact_dir)
os.mkdir(new_train_viral_dir)
os.mkdir(new_test_normal_dir)
os.mkdir(new_test_bact_dir)
os.mkdir(new_test_viral_dir)
os.mkdir(new_val_normal_dir)
os.mkdir(new_val_bact_dir)
os.mkdir(new_val_viral_dir)

base_dir = './chest_xray'
subdirs = os.listdir(base_dir)

for sd in subdirs:
    subsubdirs = os.listdir(os.path.join(base_dir, sd))
    for ssd in subsubdirs:
        fnames = os.listdir(os.path.join(base_dir, sd, ssd))
        if ssd == 'NORMAL':
            for f in fnames:
                src = os.path.join(base_dir, sd, ssd, f)
                dst = os.path.join(new_dir, sd, ssd, f)
                shutil.copy(src, dst)
        else:
            for f in fnames:
                src = os.path.join(base_dir, sd, ssd, f)
                if re.search('virus', f):
                    dst = os.path.join(new_dir, sd, 'VIRAL', f)
                elif re.search('bacteria', f):
                    dst = os.path.join(new_dir, sd, 'BACTERIAL', f)
                shutil.copy(src, dst)

