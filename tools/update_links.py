import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'project_homepage': 'Project homepage', 'code': 'Code', 'hardware': 'Hardware', 'vision_models': 'Vision models', 'image_datasets': 'Image datasets', 'paper': 'Paper'}

def render(links):
    lines = ['# Project resources', '', 'The homepage connects the paper, robot software, hardware designs, models and datasets.', '']
    for key, label in LABELS.items():
        url = links.get(key, '')
        if not isinstance(url, str):
            raise ValueError(f'{key} must be a URL string or an empty string.')
        if url and (urlparse(url).scheme not in ('https', 'http') or not urlparse(url).netloc or any(c in url for c in '\n\r<>')):
            raise ValueError(f'Invalid URL for {key}: {url}')
        lines += [f'## {label}', '', f'[{label}](<{url}>)' if url else 'Public URL has not been supplied yet.', '']
    lines += ['Edit `project-links.json` and run `python tools/update_links.py` in the code repository to refresh this page.', '']
    return '\n'.join(lines)

def main():
    parser = argparse.ArgumentParser(description='Generate resource links from project-links.json.')
    parser.add_argument('--strict', action='store_true')
    args = parser.parse_args()
    links = json.loads((ROOT / 'project-links.json').read_text(encoding='utf-8'))
    content = render(links)
    missing = [key for key in LABELS if not links.get(key)]
    if args.strict and missing:
        parser.error('Missing public URLs: ' + ', '.join(missing))
    (ROOT / 'docs/project-links.md').write_text(content, encoding='utf-8')
    hardware = ROOT / 'hardware'
    if hardware.is_dir():
        (hardware / 'docs').mkdir(exist_ok=True)
        (hardware / 'project-links.json').write_text(json.dumps(links, indent=2) + '\n', encoding='utf-8')
        (hardware / 'docs/project-links.md').write_text(content, encoding='utf-8')
    print('Resource pages updated.' + (' URLs still needed: ' + ', '.join(missing) if missing else ''))

if __name__ == '__main__':
    main()
