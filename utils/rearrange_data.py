'''This script was used to rearrange the provided dataset into one that works with pytorch's ImageFolder class.
Later, we implemented custom dataset object so this is not used anymore.'''
import pandas as pd
import os
import shutil

df = pd.read_csv('labels.csv')

# Currently the data is arrange like this:
# trainval/01aaa345-52ad-4939-8207-2d39c11acfdc/dddd_image.jpg

# We want to rearrange it like this:
# trainval/0/dddd_image.jpg
# trainval/1/dddd_image.jpg
# trainval/2/dddd_image.jpg
# trainval/3/dddd_image.jpg
guid_paths = df['guid/image'].to_list()
labels = df['label'].to_list()
print(guid_paths[:5])
print(labels[:5])
print(guid_paths.index(guid_paths[0]))
print(labels[guid_paths.index(guid_paths[0])])

for p in guid_paths:
    # search what label guid/dddd_image is
    # move that image to the label folder
    imagepath = 'trainval/' + p + '_image.jpg'
    #imagepath = 'test/' + p + '_image.jpg'
    lab = labels[guid_paths.index(p)]
    print('from: ' + imagepath)
    print('to: ' + 'py_train/{}/{}.jpg'.format(lab, p.replace('/','')))

    shutil.move(imagepath, 'py_train/{}/{}.jpg'.format(lab, p.replace('/','')))
