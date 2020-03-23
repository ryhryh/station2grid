import gdown
import os

# download data
url = 'https://drive.google.com/uc?id=111-2f3i4gYIXuB9WZT4CQ5qnspe4eujS'
output = 'data.tar'
gdown.download(url, output, quiet=False)


# unzip epa, air, sat files
# remove data.tar
os.system('tar xvf data.tar')
os.system('rm data.tar')