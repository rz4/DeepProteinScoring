import glob, os, csv

target = 'T0895'
sfid_list = ['01-01_0c1', '01-01_1c1', '01-01_2c1', '01-02_0c1', '01-02_1c1', '01-02_2c1', '01-02_3c1', '01-02_4c1']

def rename(dir, pattern, sfid):
    files_changed = 0

    for pathAndFilename in glob.iglob(os.path.join(dir, pattern)):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        title = target + '-' + sfid + '_' + '_'.join(title.split('_')[:])
        #os.rename(pathAndFilename, os.path.join(dir, title + ext))
        print(title + ext)
        files_changed += 1

    return files_changed

for sfid in sfid_list:
    print('In {}, files changed: {}'.format(sfid, rename(sfid + r'/pdbs',
        r'*.pdb', sfid)))

