# double-wilson
Exploring the idea of the Double-Wilson distribution

## Installation

```
pip install -e .
```

### Note on conda environments and `wilson.yml`
I (Dennis) pushed a file `wilson.yml` with the dependencies (as best I can tell!) for the notebooks here. Creating a new environment using this file can be done via:
```
conda env create --file wilson.yml
```
If either I'm missing packages or new packages become required, it seems like the best workflow is
1. Install the additional necessary packages to your local copy of the environment.
2. Export the updated environment via
```
conda env export --name wilson > wilson.yml
```
3. Push the updated `wilson.yml` file to github.  
  
Then, for another user, you can make sure you're up-to-date by calling
```
conda remove --name wilson --all
conda env create --file wilson.yml
```
This is maybe all overkill (and also kind of slow to do every time, takes a minute or two), but seems like it might avoid some confusion/bugs?
