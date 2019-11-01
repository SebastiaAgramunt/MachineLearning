## Running code on Docker

Install [Docker](https://www.docker.com/get-started) in your computer. Try to get your docker version on the command line

```sh 
docker --version
```
Simply run the docker-compose
```sh
docker-compose up
```
And open the notebook in your [browser](http://localhost:8888).

## Running code using Pyenv

Make sure you have installed [pyenv](https://github.com/pyenv/pyenv) and 
[pyenv-virtualenvwrapper](https://github.com/pyenv/pyenv-virtualenvwrapper). Run the following command to create a 
virtualenviroment called ml

```bash
mkvirutalenv ml
```

The local Python version is specified in ```.python-version``` file. Then install all the required packages

```bash
pip install -r requirements.txt
```

Now install the kernel in jupyter

```bash
$(which python) -m ipykernel install --name=ml
```

check the instructions [here](https://gist.github.com/SebastiaAgramunt/5185ccf8637e69f611bd1217a98289b2). Then you can launch the notebook by typing

```bash
jupyter notebook
```
