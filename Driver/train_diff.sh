#!/bin/sh

runid_start=36
runid_end=41

for (( r=${runid_start}; r <= ${runid_end}; r++ )); do
  # Append 'r' with zeros to make it a 5-digit number
  printf -v padded_r "%04d" "$r"
  qsub -N "DiffTr${padded_r}" "DiffTr" "${r}"
  sleep 3
done
