# SD-XRAY

Step by step to use poetry library to build the python project:

- Install python 3.10 (the specified version in pyproject.toml is 3.10.8, but change it if needed)
- Run py -3.10 -m pip install poetry and the following commands on the root folder of this repository
- Execute the command py -3.10 -m poetry config virtualenvs.in-project true
- Execute the command py -3.10 -m poetry install.
- Now activate the created virtual environment with .\.venv\Scripts\activate
- If any change was applied to pyproject.toml file, delete the poetry.lock file and proceed with the steps mentioned above normally