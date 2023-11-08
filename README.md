# Graphene - BC

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

## Git LFS

+ GitHub repository settings
  + `Archives` → `Include Git LFS objects in archives`

+ Install Git-LFS extension: `git lfs install`
  + Initialization: `git lfs install`

+ Specify the file types to be managed by LFS
  + Create a file named  `.gitattributes` in the root directory of your repository
  + Specify the file types that need to be managed by LFS in this file using either the file extension or file path
    + for zip files: `*.zip filter=lfs diff=lfs merge=lfs -text`
    + for files without filetype: `large-file filter=lfs diff=lfs merge=lfs -text`
  + List the large files: `git lfs ls-files`

+ Track and commit LFS files
  + Track
    + for zip files: `git lfs track "*.zip"`
    + for files without filetype: `git lfs track "large-file"`
    + for the specific file: `git lfs track large-file_directory`
  + Add the LFS list
    + LFS attributes: `git add .gitattributes`
    + LFS files: `git add large_file_directory`
+ Commit the changes: `git commit -m "Commit message"`
  + Push the change: `git push`
  + Push to the origin branch: `git push origin main`

+ Pull the repository
  + Downloads the LFS files from the Git LFS server: `git lfs fetch`
  + Checks out these files from the local LFS repository: `git lfs checkout`
