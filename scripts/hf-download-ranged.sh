#!/bin/bash
# Fast, resumable download of big files from the Hugging Face Hub: one connection gets ~1 MiB/s from the CDN, so every file is fetched as 128 MiB
# ranges, 24 at a time, straight into a pre-sized file (a .done marker per finished range; run it again to resume or to fill a failed range).
#   HF_TOKEN=hf_... scripts/hf-download-ranged.sh REPO DEST_DIR FILE...        (the token is optional for public repositories)
# The aggregate rate on jai tops out at ~25 - 30 MB/s however many connections; verify the result against the Hub's SHA-256
# (curl -s https://huggingface.co/api/models/REPO?blobs=true | jq '.siblings[] | [.rfilename, .lfs.sha256]') with sha256sum -c.
set -u
REPO=$1; D=$2; shift 2
R=https://huggingface.co/$REPO/resolve/main
PART=$((128 * 1024 * 1024))
mkdir -p "$D/.parts"
K="$D/.parts/auth.curl"
umask 077
: > "$K"
[ -n "${HF_TOKEN:-}" ] && printf 'header = "Authorization: Bearer %s"\n' "$HF_TOKEN" > "$K"

part() {
	f=$1; n=$2; size=$3
	a=$((n * PART)); b=$((a + PART - 1)); [ $b -ge $size ] && b=$((size - 1))
	m="$D/.parts/$f.$n.done"
	[ -e "$m" ] && exit 0
	for try in 1 2 3 4 5 6; do
		curl -sS -L -f -K "$D/.parts/auth.curl" -r $a-$b "$R/$f" | dd of="$D/$f" bs=1M seek=$((a / 1048576)) conv=notrunc status=none
		[ ${PIPESTATUS[0]} -eq 0 ] && { touch "$m"; exit 0; }
		sleep 90                                   # (the Hub rate-limits fast retries)
	done
	exit 1
}
export -f part; export R D PART

for f in "$@"; do
	size=$(curl -sIL -K "$K" "$R/$f" | tr -d '\r' | awk 'tolower($1) == "content-length:" {s = $2} END {print s}')
	[ -z "$size" ] && { echo "no size for $f"; continue; }
	[ -e "$D/$f" ] || truncate -s "$size" "$D/$f"
	echo "$f $size"
	seq 0 $(((size + PART - 1) / PART - 1)) | xargs -P 24 -I{} bash -c "part $f {} $size"
done

rm -f "$K"                                          # (do not leave the token on disk)
echo done
