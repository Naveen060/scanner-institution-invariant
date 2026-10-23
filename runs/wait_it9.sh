#!/bin/bash
cd /home/claude/sidl
until [ -f runs/it9_done ]; do
  n=$(find runs/conf_lso -name extra_H.npz 2>/dev/null | wc -l)
  l=$(cat runs/conf_*/results.jsonl 2>/dev/null | wc -l)
  if [ "$n" -ge 72 ] && [ "$l" -ge 72 ]; then touch runs/it9_done; fi
  sleep 30
done
