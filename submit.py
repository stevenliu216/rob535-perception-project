from glob import glob
import string

test_dir = glob('test/*/*_image.jpg')
new_test_dir = [ s.strip('test/') for s in test_dir ]
print(new_test_dir)
new_test_dir2 = [ s.strip('_image.jpg') for s in new_test_dir ]
new_test_dir3 = [ s + ',1' for s in new_test_dir2 ]

# write to file
with open('output.csv', 'w') as f:
    f.write('guid/image,label\n')
    for s in new_test_dir3:
        f.write('%s\n' % s)
