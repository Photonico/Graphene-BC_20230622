#!/bin/bash

# Change directory to the location of the script
cd "$(dirname "$0")"

# List of files to search for
files_to_search=(
  "CHG"
  "CHGCAR"
  "DOSCAR"
  "EIGENVAL"
  "IBZKPT"
  "INCAR"
  "KPOINTS"
  "OUTCAR"
  "output.txt"
  "PCDAT"
  "POTCAR"
  "REPORT"
  "vasp_nci.sh"
  "vasp_usyd.sh"
  "vasp.log"
  "vasp.out"
  "vaspout.h5"
  "WAVECAR"
  "XDATCAR"
)

# Start searching
echo "Searching for files in: $(pwd) and its subdirectories"
for file in "${files_to_search[@]}"; do
  find . -type f -name "$file" 2>/dev/null
done
