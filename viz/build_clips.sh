#!/usr/bin/env bash
# Render the pdfr site clips: PNG masters -> per-shot intermediates -> dissolves -> final encode.
# usage: viz/build_clips.sh <masters_dir> <delivery_dir> '<post-processing params json>' [score_floor]
set -euo pipefail
cd "$(dirname "$0")/.."
M=$1; D=$2; PARAMS=$3; FLOOR=${4:-0.9}; KEY=pdfr
mkdir -p "$M" "$D"
.venv/bin/python viz/animate.py --shots viz/pdfr_shots.json --out "$M" --score_floor "$FLOOR" --fps 30 --params "$PARAMS"
enc_mid() { ffmpeg -v error -y -framerate 30 -i "$M/$1/%05d.png" -c:v libx264 -crf 10 -preset medium -pix_fmt yuv420p "$M/$1.mp4"; }
final() {
  ffmpeg -v error -y -i "$1" -an \
    -vf "scale=1200:800:flags=lanczos,fps=30,format=yuv420p" \
    -c:v libx264 -profile:v high -preset slow -crf 23 -maxrate 2M -bufsize 4M \
    -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv \
    -movflags +faststart "$2"
}
for s in c1_day c2a_wakeup c2b_week c3a_onset c3b_twodays; do enc_mid $s; done
ffmpeg -v error -y -i "$M/c1_day.mp4" -t 6 -c:v libx264 -crf 10 "$M/clip1.mp4"
final "$M/clip1.mp4" "$D/$KEY-01.mp4"
for c in 2 3; do
  a=$(ls -d "$M"/c${c}a_*/ | xargs -n1 basename); b=$(ls -d "$M"/c${c}b_*/ | xargs -n1 basename)
  ffmpeg -v error -y -i "$M/$a.mp4" -i "$M/$b.mp4" \
    -filter_complex "[0][1]xfade=transition=fade:duration=0.25:offset=3.0,format=yuv420p" -an \
    -c:v libx264 -crf 10 "$M/clip$c.mp4"
  final "$M/clip$c.mp4" "$D/$KEY-0$c.mp4"
done
ffmpeg -v error -y -i "$D/$KEY-01.mp4" -frames:v 1 -q:v 3 "$D/$KEY-poster.jpg"

# wall cut: every shot in order (the 24 h scroll at full 12 s), same dissolves, CRF 18
ffmpeg -v error -y -i "$M/c1_day.mp4" -i "$M/c2a_wakeup.mp4" -i "$M/c2b_week.mp4" -i "$M/c3a_onset.mp4" -i "$M/c3b_twodays.mp4" \
  -filter_complex "[0][1]xfade=transition=fade:duration=0.25:offset=11.75[a];[a][2]xfade=transition=fade:duration=0.25:offset=14.75[b];[b][3]xfade=transition=fade:duration=0.25:offset=17.5[c];[c][4]xfade=transition=fade:duration=0.25:offset=20.5,scale=1200:800:flags=lanczos,format=yuv420p" \
  -an -c:v libx264 -profile:v high -preset slow -crf 18 -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv \
  -movflags +faststart "$M/$KEY-wall.mp4"
