#!/usr/bin/env python3
import re
from urllib.parse import quote as encode_uri

content = open('DS_Intuition.md').read()
IMG_ROOT_SM = 'https://latex.codecogs.com/gif.latex?' + encode_uri('\dpi{85}')
IMG_ROOT_L = 'https://latex.codecogs.com/gif.latex?' + encode_uri('\dpi{120}')

content = re.sub(
    r'\$\$(.+?)\$\$',
    lambda x: f'''<p align="center"><img src="{IMG_ROOT_L}{encode_uri(x.group(1))}"></p>''',
    content)

content = re.sub(
    r'\$([^\$\n\r]+)\$',
    lambda x: f'![equation]({IMG_ROOT_SM}{encode_uri(x.group(1))})',
    content)

open('README.md', 'w').write(content)
