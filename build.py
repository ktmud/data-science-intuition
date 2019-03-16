#!/usr/bin/env python3
import re
import os

from urllib.parse import quote as encode_uri

content = open('DS_Intuition.md').read()
IMG_ROOT_SM = 'https://latex.codecogs.com/gif.latex?' + encode_uri('\dpi{100}')
IMG_ROOT_L = 'https://latex.codecogs.com/gif.latex?' + encode_uri('\dpi{130}')

content = re.sub(
    r'\$\$(.+?)\$\$',
    lambda x: f'''<p align="center"><img alt="" src="{IMG_ROOT_L}{encode_uri(x.group(1))}"></p>''',
    content)

content = re.sub(
    r'\$([^\$\n\r]+)\$',
    lambda x: f'![-]({IMG_ROOT_SM}{encode_uri(x.group(1))})',
    content)

open('README.md', 'w').write(content)
os.system('./gh-md-toc --insert README.md')
os.system('rm README.md.*')
