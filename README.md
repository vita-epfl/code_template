### code_template
A repository displaying a possible code structure suitable for Slurm


#### Poetry - Dependency Management
Install Poetry 
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Initialise it once in your repository
```bash
poetry init
```

Add requirements
```bash
poetry add {name_of_package}
```

Activate the virtual environment
```bash
poetry shell
```

Install requirements if needed
```bash
poetry install
```

#### Hydra - Config File Management
```bash
pip install hydra-core --upgrade
```

#### Submitit - Slurm job Management
```bash
poetry add submitit
```

#### Poetry-Keyring Bug 
To fix this reported bug, add the following to your .bashrc / .zhsrc
```bash
export PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring"
```
