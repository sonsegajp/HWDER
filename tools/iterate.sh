#!/bin/sh
# Boot loop: run the game; if it reaches an unrecompiled address, add it to data/extra_entries.txt,
# regenerate + rebuild (incremental), and repeat. Usage: tools/iterate.sh [max_iterations] [seconds]
cd "$(dirname "$0")/.."
max=${1:-10}; secs=${2:-60}
for i in $(seq 1 $max); do
  rm -f build/release/extra_entries.txt
  (cd build/release && HWDER_LOG=iter.log HWDER_EXEFS=../../data/exefs HWDER_ROMFS=../../data/romfs timeout $secs ./hwder.exe > run.txt 2>&1)
  if [ ! -s build/release/extra_entries.txt ]; then echo "iteration $i: no new entry points"; exit 0; fi
  touch data/extra_entries.txt
  cat build/release/extra_entries.txt >> data/extra_entries.txt
  sort -u data/extra_entries.txt -o data/extra_entries.txt
  echo "iteration $i: +$(wc -l < build/release/extra_entries.txt) entry points"
  (cd tools && python -m recomp.gen_main > /dev/null) && python tools/build.py > /dev/null 2>&1 || { echo build failed; exit 1; }
done
