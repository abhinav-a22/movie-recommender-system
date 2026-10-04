"""
Script to automatically verify, download, and extract the MovieLens 1M dataset.
Guarantees offline readiness after initial download.
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, REQUIRED_DATA_FILES, DATASET_URL


def check_dataset_exists() -> bool:
    """Verify that all required .dat files exist and are non-empty."""
    if not DATA_DIR.exists():
        return False
    for req_file in REQUIRED_DATA_FILES:
        if not req_file.exists() or req_file.stat().st_size == 0:
            return False
    return True


def download_and_extract(force: bool = False) -> bool:
    """
    Downloads MovieLens 1M zip if missing or force=True, extracts .dat files,
    validates existence, and prints status.
    """
    if not force and check_dataset_exists():
        print("✓ MovieLens 1M dataset already exists locally in:", DATA_DIR)
        for req_file in REQUIRED_DATA_FILES:
            size_mb = req_file.stat().st_size / (1024 * 1024)
            print(f"  - {req_file.name}: {size_mb:.2f} MB")
        return True

    print("============================================================")
    print("           DOWNLOADING MOVIELENS 1M DATASET                 ")
    print("============================================================")
    print(f"Target directory: {DATA_DIR}")
    print(f"Dataset URL: {DATASET_URL}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR.parent / "ml-1m.zip"

    # Download with streaming
    print("Downloading dataset archive...")
    try:
        import requests
        response = requests.get(DATASET_URL, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        downloaded = 0
        chunk_size = 1024 * 64
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:5.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)", end="", flush=True)
                    else:
                        print(f"\rDownloaded: {downloaded / (1024*1024):.1f} MB", end="", flush=True)
        print("\n✓ Download completed successfully.")
    except Exception as e:
        print(f"\nRequests download failed ({e}). Attempting fallback via urllib...")
        import urllib.request
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        try:
            req = urllib.request.Request(DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=30) as res, open(zip_path, "wb") as out_f:
                shutil.copyfileobj(res, out_f)
            print("✓ Urllib download completed successfully.")
        except Exception as err2:
            print(f"✗ Failed to download dataset: {err2}")
            if zip_path.exists():
                zip_path.unlink()
            return False

    # Extract files
    print(f"Extracting archive '{zip_path.name}'...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                filename = os.path.basename(member)
                if filename in ["users.dat", "movies.dat", "ratings.dat", "README"]:
                    source = zf.open(member)
                    target = open(DATA_DIR / filename, "wb")
                    with source, target:
                        shutil.copyfileobj(source, target)
        print("✓ Extraction complete.")
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False
    finally:
        # Clean up temporary zip file
        if zip_path.exists():
            zip_path.unlink()

    # Validate required files
    if check_dataset_exists():
        print("✓ Validation passed! All required files are present:")
        for req_file in REQUIRED_DATA_FILES:
            size_mb = req_file.stat().st_size / (1024 * 1024)
            print(f"  - {req_file.name}: {size_mb:.2f} MB")
        return True
    else:
        print("✗ Validation failed: One or more required .dat files are missing.")
        return False


if __name__ == "__main__":
    success = download_and_extract()
    sys.exit(0 if success else 1)
