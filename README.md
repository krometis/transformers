# Introduction to Transformers
Slides and code from my Introduction to Transformers lecture on January 28, 2026

Contents:
- Slides (generated from Markdown via [marp](https://marp.app/))
- Python code

## One-time Setup
To run the notebooks, you will need to have the right packages installed.

_Note_: If running on [ARC](https://arc.vt.edu) resources, it may be helpful to run 
```
module reset; module load Miniforge3/25.11.0-1
```
prior to other commands to get access to a newer version of Python.

### Recommended: Create a virtual environment
Linux/Mac:
```bash
#create the virtual environment
python -m venv transformers
#activate it
source transformers/bin/activate
#install required packages
pip install -r requirements.txt
#add environment to jupyter
#(once done, may require restarting jupyter instance for environment to show up as kernel choice)
python -m ipykernel install --user --name=transformers --display-name="Python (Transformers)"
#download trained model
python -m spacy download en_core_web_lg
#deactivate the virtual environment
deactivate
```

Windows Command Prompt:
```bash
#create the virtual environment
python -m venv transformers
#activate it
transformers\Scripts\activate.bat
#install required packages
pip install -r requirements.txt
#add environment to jupyter
#(once done, may require restarting jupyter instance for environment to show up as kernel choice)
python -m ipykernel install --user --name=transformers --display-name="Python (Transformers)"
#download trained model
python -m spacy download en_core_web_lg
#deactivate the virtual environment
deactivate
```

### Basic
Just install the requirements:
```bash
pip install -r requirements.txt
```
And then download the model:
```bash
python -m spacy download en_core_web_lg
```


## Running
To run the notebooks:

1. Load the virtual environment, if you created one:

```bash
source transformers/bin/activate   #Linux/Mac
transformers\Scripts\activate.bat  #Windows Command Prompt
```

2. Start the notebook
```bash
jupyter notebook
```

3. Deactivate the virtual environment, if you're using one
```bash
deactivate
```
