# SD-XRAY

Step by step to use poetry library to build the python project:

- Install python 3.10 (the specified version in pyproject.toml is 3.10.9, but change it if needed)
- Run the command below to install poetry (if it is not already installed) and the following commands on the root folder of this repository to setup the environment
    ``` sh
    py -3.10 -m pip install poetry
    ```
- Execute the command: 
    ``` sh
    py -3.10 -m poetry config virtualenvs.in-project true
    ```
- Followed by:
    ``` sd
    py -3.10 -m poetry install
    ```
- If any change was applied to pyproject.toml file, delete the poetry.lock file and proceed with the steps mentioned above normally