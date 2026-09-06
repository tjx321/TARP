#!/bin/bash
# Download the precomputed TARP feature caches (~22 GB, fp16) from Google Drive
# into ./caches/.
#
# The archive is shared as a public Google Drive folder:
#   https://drive.google.com/drive/folders/1YgExR-xJdEqVf3u43hhGiiqFe9I57gHm
# Layout inside: TARP_caches/{rn50,rn101,vitb16,vitb32}/seed{1,2,3}/<dataset>/
set -e
cd "$(dirname "$0")/.."

GDRIVE_FOLDER="https://drive.google.com/drive/folders/1YgExR-xJdEqVf3u43hhGiiqFe9I57gHm"

if ! command -v gdown >/dev/null 2>&1; then
  echo "gdown not found. Installing with pip ..."
  pip install -q gdown
fi

echo "Downloading TARP caches (~22 GB) from Google Drive into ./caches ..."
echo "Folder: $GDRIVE_FOLDER"

# Download the shared folder into a temp dir, then move the backbone trees into ./caches.
TMPDIR=$(mktemp -d)
gdown --folder "$GDRIVE_FOLDER" -O "$TMPDIR" --remaining-ok

# The shared folder may nest under "TARP_caches" or directly hold the backbone dirs.
SRC="$TMPDIR"
[ -d "$TMPDIR/TARP_caches" ] && SRC="$TMPDIR/TARP_caches"

mkdir -p caches
for bb in rn50 rn101 vitb16 vitb32; do
  if [ -d "$SRC/$bb" ]; then
    echo "Merging $bb ..."
    cp -r "$SRC/$bb" caches/
  fi
done

rm -rf "$TMPDIR"
echo "Done. Expected layout: caches/{rn50,rn101,vitb16,vitb32}/seed{1,2,3}/<dataset>/"
