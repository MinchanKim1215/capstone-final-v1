"""Download the user-provided Google Drive file, without executing its contents."""
from pathlib import Path
from urllib.request import build_opener, HTTPCookieProcessor, Request
from urllib.parse import urlencode, urlparse, urljoin, unquote
from http.cookiejar import CookieJar
from html.parser import HTMLParser
from datetime import datetime
import hashlib
import json
import os
import re
import time
import unicodedata
import zipfile

FILE_ID = '1K9nMNNea6FdmP_TZ5LlgPAbEXgHdW7li'
OPS = Path(__file__).resolve().parent
DOWNLOADS = Path('C:/Users/mckim/Downloads')
RECORD = OPS / ('drive_download_'+FILE_ID+'.json')


def emit(**data):
    data['updated_at'] = datetime.now().astimezone().isoformat(timespec='seconds')
    RECORD.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(data, ensure_ascii=False), flush=True)


class Forms(HTMLParser):
    def __init__(self):
        super().__init__()
        self.forms = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'form':
            self.current = dict(action=a.get('action', ''), method=a.get('method', 'get').lower(), fields={})
        elif tag == 'input' and self.current is not None and a.get('name'):
            self.current['fields'][a['name']] = a.get('value', '')

    def handle_endtag(self, tag):
        if tag == 'form' and self.current is not None:
            self.forms.append(self.current)
            self.current = None


def safe_name(headers):
    disposition = headers.get('Content-Disposition', '')
    encoded = re.search(r"filename\*\s*=\s*UTF-8''([^;]+)", disposition, re.I)
    name = unquote(encoded.group(1).strip('" ')) if encoded else headers.get_filename()
    if name:
        try:
            name = name.encode('latin-1').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
        name = unicodedata.normalize('NFC', name)
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name or ('google_drive_'+FILE_ID+'.bin')).strip(' .')
    if not name or Path(name).stem.upper() in {'CON','PRN','AUX','NUL',*[f'COM{i}' for i in range(1,10)],*[f'LPT{i}' for i in range(1,10)]}:
        name = 'google_drive_'+FILE_ID+'.bin'
    return name


def destination(name):
    p = DOWNLOADS/name
    counter = 1
    while p.exists() or p.with_name(p.name+'.part').exists():
        p = DOWNLOADS/(Path(name).stem+f' ({counter})'+Path(name).suffix)
        counter += 1
    return p


def main():
    if os.name == 'nt':
        import ctypes
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000)
    emit(state='connecting', file_id=FILE_ID, destination_folder=str(DOWNLOADS))
    opener = build_opener(HTTPCookieProcessor(CookieJar()))
    url = 'https://drive.usercontent.google.com/download?'+urlencode(dict(id=FILE_ID, export='download', confirm='t'))
    response = None
    for attempt in range(4):
        response = opener.open(Request(url, headers={'User-Agent':'Mozilla/5.0', 'Accept-Encoding':'identity'}), timeout=120)
        if response.headers.get_content_type() != 'text/html' or 'attachment' in response.headers.get('Content-Disposition', '').lower():
            break
        content = response.read(2_000_000).decode('utf-8', errors='replace')
        current_url = response.geturl()
        response.close()
        forms = Forms()
        forms.feed(content)
        choices = [form for form in forms.forms if form['fields'].get('id') == FILE_ID
                   and urlparse(urljoin(current_url, form['action'])).hostname in {'drive.google.com', 'drive.usercontent.google.com'}
                   and form['method'] == 'get']
        if not choices:
            if 'accounts.google.com' in current_url or 'request access' in content.lower() or '액세스 요청' in content:
                raise RuntimeError('Drive requires account sign-in or file-owner access approval; no file downloaded.')
            raise RuntimeError('Drive returned an HTML page without an authorized file-download form; no file downloaded.')
        form = choices[0]
        url = urljoin(current_url, form['action'])+'?'+urlencode(form['fields'])
        emit(state='confirming_drive_download', file_id=FILE_ID, attempt=attempt+1)
    else:
        raise RuntimeError('Drive download confirmation did not resolve.')
    name = safe_name(response.headers)
    target = destination(name)
    partial = target.with_name(target.name+'.part')
    expected = response.headers.get('Content-Length')
    expected = int(expected) if expected and expected.isdigit() else None
    emit(state='downloading', filename=target.name, path=str(target), expected_bytes=expected, downloaded_bytes=0)
    digest = hashlib.sha256()
    count = 0
    began = last = time.monotonic()
    with response, partial.open('xb') as output:
        while True:
            chunk = response.read(1024*1024)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
            count += len(chunk)
            if time.monotonic()-last >= 10:
                emit(state='downloading', filename=target.name, path=str(target), expected_bytes=expected,
                     downloaded_bytes=count, elapsed_s=round(time.monotonic()-began, 1))
                last = time.monotonic()
        output.flush()
        os.fsync(output.fileno())
    if expected is not None and expected != count:
        raise RuntimeError(f'Incomplete transfer: expected {expected}, received {count}; partial file preserved.')
    if count == 0:
        raise RuntimeError('Drive sent an empty file; partial preserved.')
    zip_summary = None
    if target.suffix.lower() == '.zip':
        if not zipfile.is_zipfile(partial):
            raise RuntimeError('A ZIP filename was returned, but downloaded data is not a ZIP archive; partial preserved.')
        with zipfile.ZipFile(partial) as archive:
            files = [info for info in archive.infolist() if not info.is_dir()]
            zip_summary = dict(files=len(files), uncompressed_bytes=sum(info.file_size for info in files),
                               central_directory_valid=True, full_crc_check_performed=False)
    partial.rename(target)
    emit(state='completed', filename=target.name, path=str(target), bytes=count, sha256=digest.hexdigest(),
         elapsed_s=round(time.monotonic()-began, 1), zip_summary=zip_summary, executed_or_extracted=False)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        emit(state='failed', file_id=FILE_ID, error=str(exc))
        raise
