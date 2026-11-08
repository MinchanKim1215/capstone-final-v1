"""Verify delivered bytes without changing models or opening final-test scenarios."""
import argparse,hashlib,json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--full',action='store_true',help='Also verify the delivered mutable evidence; use immediately after extraction')
    parser.add_argument('--optional-evidence',action='store_true',help='Verify all optional raw evidence after extracting the companion archive')
    args=parser.parse_args();root=Path(__file__).resolve().parent
    manifest=json.loads((root/'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
    paths=dict(manifest['immutable'])
    if args.full:paths.update(manifest['delivered_evidence'])
    if args.optional_evidence:paths.update(manifest['optional_evidence'])
    errors=[]
    for name,want in paths.items():
        path=root/name
        if not path.is_file():errors.append('missing: '+name);continue
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
        if digest.hexdigest()!=want:errors.append('changed: '+name)
    result=dict(passed=not errors,checked=len(paths),format=manifest['format'],full=args.full,errors=errors,
        note='Default verifies immutable delivery files. --full is intended for the original extracted package before further training.')
    print(json.dumps(result,indent=2));return 0 if not errors else 1


if __name__=='__main__':raise SystemExit(main())
