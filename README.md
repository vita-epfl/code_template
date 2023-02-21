### code_template
A repository displaying a possible code structure suitable for Slurm


#### Scitas - Python versioning

* Option 1: 
```bash
module load {python_version}
```

* Option 2: Install any version of python using spack as a module then load it.
Example:
Install spack cf. https://spack.readthedocs.io/en/latest/getting_started.html
Choose the version of python to install
```bash
srun -p build -t 01:00:00 --gres gpu:1 --pty spack install python@3.8.14
```
Create the module:
```bash
module use <spack_install_dir>/share/spack/modules/linux-rhel8-skylake_avx512/
```

Load the module (check the right name using spider)

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
