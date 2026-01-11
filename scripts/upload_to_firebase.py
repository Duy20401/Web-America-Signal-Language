#!/usr/bin/env python3
"""
Upload local letter images and word videos to Firebase Storage.

Usage examples:
  python scripts/upload_to_firebase.py --service-account asl_web/firebase-service-account.json \
      --project asl-learning-app-b53e6 \
      --letters-path learning/static/learning/images/ASL_Alphabet \
      --words-path learning/static/learning/videos --public

Flags:
  --dry-run   : don't actually upload, only print what would be uploaded
  --public    : make uploaded blobs public (call blob.make_public())

This script uses the `firebase-admin` SDK. Install with:
  python -m pip install firebase-admin

Note: ensure the service account JSON exists at the given path (project includes `asl_web/firebase-service-account.json`).
"""
import argparse
import os
import sys
from pathlib import Path

try:
    import firebase_admin
    from firebase_admin import credentials, initialize_app, storage
except Exception as e:
    print("Missing dependency: firebase-admin. Install with: python -m pip install firebase-admin")
    raise


def find_files(path, exts):
    p = Path(path)
    if not p.exists():
        return []
    files = []
    for ext in exts:
        files.extend(list(p.rglob(f'*{ext}')))
    # sort for deterministic order
    files.sort()
    return files


def upload_blob(bucket, local_path: Path, dest_path: str, public=False, dry_run=False):
    print(f"Uploading {local_path} -> {dest_path}")
    if dry_run:
        return
    blob = bucket.blob(dest_path)
    blob.upload_from_filename(str(local_path))
    if public:
        try:
            blob.make_public()
            print(f"  -> public url: {blob.public_url}")
        except Exception as e:
            print(f"  -> uploaded but failed to make public: {e}")


def main():
    parser = argparse.ArgumentParser(description='Upload letters and words to Firebase Storage')
    parser.add_argument('--service-account', required=True, help='Path to service account JSON')
    parser.add_argument('--project', required=True, help='Firebase project id')
    parser.add_argument('--letters-path', help='Local folder containing letter images (A.jpg, B.jpg, ...)', default='learning/static/learning/images/ASL_Alphabet')
    parser.add_argument('--words-path', help='Local folder containing word video files or thumbnails', default='learning/static/learning/videos')
    parser.add_argument('--letters-dest', help='Destination folder in storage for letters', default='letters')
    parser.add_argument('--words-dest', help='Destination folder in storage for words', default='words')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be uploaded without uploading')
    parser.add_argument('--public', action='store_true', help='Make uploaded files publicly readable')
    args = parser.parse_args()

    sa_path = Path(args.service_account)
    if not sa_path.exists():
        print(f"Service account file not found: {sa_path}")
        sys.exit(2)

    cred = credentials.Certificate(str(sa_path))
    app = initialize_app(cred, {'storageBucket': f'{args.project}.appspot.com'})
    bucket = storage.bucket()

    # Upload letters
    print('\n=== Letters ===')
    letter_exts = ['.jpg', '.jpeg', '.png', '.webp']
    letters = find_files(args.letters_path, letter_exts)
    if not letters:
        print(f'No letter images found in {args.letters_path} (checked {letter_exts})')
    else:
        for f in letters:
            name = f.name
            # set destination as letters/<name>
            dest = f"{args.letters_dest}/{name}"
            upload_blob(bucket, f, dest, public=args.public, dry_run=args.dry_run)

    # Upload words (videos and thumbnails)
    print('\n=== Words ===')
    word_exts = ['.mp4', '.webm', '.jpg', '.jpeg', '.png', '.webp']
    words = find_files(args.words_path, word_exts)
    if not words:
        print(f'No word media found in {args.words_path} (checked {word_exts})')
    else:
        for f in words:
            name = f.name
            dest = f"{args.words_dest}/{name}"
            upload_blob(bucket, f, dest, public=args.public, dry_run=args.dry_run)

    print('\nDone.')


if __name__ == '__main__':
    main()
