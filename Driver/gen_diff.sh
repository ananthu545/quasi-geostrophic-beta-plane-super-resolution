#!/bin/sh

runid_start=41
runid_end=41

genid_start=1
genid_end=4

for (( r=${runid_start}; r <= ${runid_end}; r++ )); do
  # Zero-pad 'r' to 4 digits (e.g., 0035)
  printf -v padded_r "%04d" "$r"

  for (( g=${genid_start}; g <= ${genid_end}; g++ )); do
    # Zero-pad 'g' to 04 digits (adjust width if you prefer)
    printf -v padded_g "%04d" "$g"

    qsub -N "DG${padded_r}_${padded_g}" "DiffGen" "${r}" "${g}"
    sleep 3
  done
done
