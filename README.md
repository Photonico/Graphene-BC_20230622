# Graphene - BC

This project has been publicated: <https://doi.org/10.3390/nano14201659>

## Matter

### Monolayers

1. g BC3 (BC₃)

2. Graphene

3. α' Borophe

4. B4C3 (B₄C₃)

### bilayers

1. Graphene - g BC3 (BC₃)

2. Graphene - α' Borophene

3. Graphene - B4C3 (B₄C₃)

## Calculation steps

+ Calculating the self-consistent requires the CONTCAR file. This should be renamed as POSCAR using the following command:  
  `mv CONTCAR POSCAR`

+ Calculating the band structure necessitates the initial charge density file (CHGCAR):  
  `cp ../< >/CHGCAR`

+ Calculating the Projected Density of States (PDOS) also requires the initial charge density file (CHGCAR):  
  `cp ../< >/CHGCAR`

+ In order to calculate optical properties, both the WAVECAR file and the initial charge density file are needed:  
  `cp ../< >/WAVECAR CHGCAR`

## Figure settings

+ Figure size
  + dpi: 196
  + 1×1 figure: `(9.6, 6.4)`
  + 1×2 figure: `(14.4, 6.4)`
  + 2×1 figure: `(9.6, 10.2)`
  + 2×2 figure: `(14.4, 10.2)`

+ Font size (Single figure)
  + Title: 18
  + Label: 12

+ Font size (Multiple figures)
  + Super title: 18
  + Title: 16
  + Label: 12

## Python virtual environment

+ Create:
  + Python: `python -m venv <env>`
  + Conda: `conda create --name <env>`
  
+ Initialize shell:
  + ZSH: `conda init zsh`

+ Active:
  + Python: `source <env>/bin/activate`
  + Conda: `conda activate <env>`

+ Deactivate
  + in Unix: `conda deactivate`
  + in Win: `conda deactivate`
  
+ Delete:
  + Python: `rm -r <env>`
  + Conda: `conda env remove --name <env>`

## License

The source code in this `vmatplot` is released under the MIT License.

You are welcome to use, modify, distribute, and adapt the code for academic, educational, or commercial purposes, provided that the original copyright and license notice are retained.

Unless otherwise stated, this license applies to the Python source code and related scripts in this repository. Research data, manuscript text, figures, and third-party files may be subject to separate terms.


