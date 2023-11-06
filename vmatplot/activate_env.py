#### Activate Python virtual environment
# pylint: disable = C0103, C0114

import subprocess

activate_env = "conda activate myenv"

subprocess.run(activate_env, shell = True, check = True)
