"""Download the six map ZIPs listed in the user's Drive folder."""
from concurrent.futures import ThreadPoolExecutor
from http.cookiejar import CookieJar
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlparse
from urllib.request import HTTPCookieProcessor, Request, build_opener
import hashlib
import json
import os
import time
import zipfile

from training_ops.download_shared_drive import Forms
from . import ROOT

FOLDER = '1AYTlEqeR7zeRVbHlJiom2eDQMNEj_YUU'
FILES = (
    ('capstone-additional50-maps-20261031-batch2.zip', '1_Erw96Z1VZaOvTzqziG_C3Ql4mJILFKj'),
    ('capstone-additional50-maps-20261031.zip', '1WYdAUv3eVJg0uop7R7enpD8i-9mtozNk'),
    ('capstone-additional50-maps-20261101-batch2.zip', '1SF6op1sF-rYgCUcxacEba7ukMJgLHvfj'),
    ('capstone-additional50-maps-20261101.zip', '11oCHOslLsvhxjzYlGuU97L-sU6fZ7VQC'),
    ('capstone-additional50-maps-20261104-batch2.zip', '19Pm7ZPYdDMdI-ZMw_s6WtSgs0p-H3jdg'),
    ('capstone-additional50-maps-20261104.zip', '1WgGmscHWlPRrSXdesLC1VGI3ZP6_dVCD'),
)
DEST = ROOT / 'map_inputs'
MAX_BYTES = 1024**3
DRIVE_HOSTS = {'drive.google.com', 'drive.usercontent.google.com'}


def inspect(target, file_id):
    if not zipfile.is_zipfile(target):
        raise ValueError(f'Drive response is not a ZIP: {target.name}')
    with zipfile.ZipFile(target) as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        total = sum(member.file_size for member in members)
    if total > 4 * MAX_BYTES or not members:
        raise ValueError(f'Unexpected archive size: {target.name}')
    digest = hashlib.sha256()
    with target.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return dict(name=target.name, path=str(target), file_id=file_id, bytes=target.stat().st_size,
                sha256=digest.hexdigest(), members=len(members), uncompressed_bytes=total)


def response_for(file_id):
    opener = build_opener(HTTPCookieProcessor(CookieJar()))
    url = 'https://drive.usercontent.google.com/download?' + urlencode(
        {'id': file_id, 'export': 'download', 'confirm': 't'})
    for _ in range(4):
        response = opener.open(Request(url, headers={'User-Agent': 'Mozilla/5.0',
                                                    'Accept-Encoding': 'identity'}), timeout=90)
        if (response.headers.get_content_type() != 'text/html' or
                'attachment' in response.headers.get('Content-Disposition', '').lower()):
            return response
        with response:
            base = response.geturl()
            content = response.read(2_000_000).decode('utf-8', errors='replace')
        parser = Forms()
        parser.feed(content)
        forms = [form for form in parser.forms if form['fields'].get('id') == file_id
                 and form['method'] == 'get'
                 and urlparse(urljoin(base, form['action'])).hostname in DRIVE_HOSTS]
        if not forms:
            raise PermissionError(f'Drive did not provide a download for {file_id}')
        form = forms[0]
        url = urljoin(base, form['action']) + '?' + urlencode(form['fields'])
    raise RuntimeError('Drive download confirmation loop')


def download(item):
    name, file_id = item
    target = DEST / name
    if target.exists():
        return inspect(target, file_id)
    partial = target.with_suffix('.zip.part')
    for attempt in range(3):
        try:
            with response_for(file_id) as response, partial.open('wb') as stream:
                count = 0
                size = response.headers.get('Content-Length', '')
                for chunk in iter(lambda: response.read(1024**2), b''):
                    count += len(chunk)
                    if count > MAX_BYTES:
                        raise ValueError(f'Unexpected transfer size: {name}')
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
                if count == 0 or (size.isdigit() and count != int(size)):
                    raise OSError(f'Incomplete download: {name}')
            inspect(partial, file_id)
            os.replace(partial, target)
            result = inspect(target, file_id)
            print(json.dumps({key: result[key] for key in ('name', 'bytes', 'members')}), flush=True)
            return result
        except (HTTPError, URLError, OSError) as error:
            if isinstance(error, (PermissionError, HTTPError)) and getattr(error, 'code', 403) not in (429, 500, 502, 503, 504):
                raise
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(download, FILES))
    result = dict(folder_id=FOLDER, archive_count=len(files), total_bytes=sum(file['bytes'] for file in files),
                  files=files)
    temporary = DEST / 'downloads.pending.json'
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, DEST / 'downloads.json')
    print(json.dumps({key: result[key] for key in ('archive_count', 'total_bytes')}), flush=True)


if __name__ == '__main__':
    main()
