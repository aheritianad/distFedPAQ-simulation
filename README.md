# distFedPAQ-simulation

## About

You can find this repo at <https://github.com/aheritianad/distFedPAQ-simulation>.

## Installation

1. Clone this repo by
    - cloning
        >`git clone https://github.com/aheritianad/distFedPAQ-simulation.git`
    - or by download through <https://github.com/aheritianad/distFedPAQ-simulation/zipball/master> .
2. Enter into the directory  
    >`cd distFedPAQ-simulation`

3. **[OPTIONAL but recommended]** Use a virtual environment
    - Make a virtual environment `.venv` with
        > `python3 -m venv .venv`
    - Activate the virtual environment with
        > `source .venv/bin/activate`

        You can turn deactivate `.venv` anytime with `deactivate` command.

4. Install all dependencies
     > `pip3 install -r ./distFedPAQ/requirments.txt`

## Usage

### Commands

1. See help for the arguments
   > `python3 main.py --help` or `python3 main.py -h`
2. Example of a simulation command
   > `python -i main.py -n 5 -loc 10 -ext 2000 -n-ext-ave 2 -bs 12 -lr 1e-1 -tr seq -st 5`

### Arguments (to fill)

- `-n` or `--nodes` :
- `-loc`
- ...
  
### Notes

1. `-tr par` does **not** work properly (for the moment), using `seq` is then recommended.
2. In order to get more flexibility, it is recommended to run in interactive mode (i.e. `python3 -i ...`).

## Author

Heritiana Daniel Andriasolofo  

- [x] github : [aheritianad](https://github.com/aheritianad)
- [x] linkedin : [aheritianad](https://linkedin.com/in/aheritianad)
- [x] gmail : [aheritianad](mailto:aheritianad@gmail.com)
  