import hashlib


def get_md5(file_name):
    with open(file_name, mode='rb') as f:
        file_hash = hashlib.md5()
        chunk = f.read(8192)
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(8192)

        return file_hash.hexdigest()
    
    
def check_md5(file_name, md5):
    if md5 != get_md5(file_name):
        assert False, f'md5 for {file_name} does not match'
with open('data/file_hashes_small') as f:
    for line in f:
        path, md5 = line.strip().split('\t')
        print(f'Validating {path} ...')
        check_md5(path, md5)

print('Validation succeeded!')
