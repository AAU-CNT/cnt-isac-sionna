# cnt-isac-sionna

Repo for work on sionna within the ISAC sub-group.

## Structure

- ```DT-reliability``` contains the code for the
[Digital Twin-Guided Wireless Localization via Probabilistic Inverse Learning](https://arxiv.org/abs/2610.36905v1)
letter.
- ```DT-selection``` contains code for the DT assisted sensor location selection paper.
- ```scenario``` contains a mitsuba scene for the location outside the kantine at FBV7.

## Setting up python environment.

Currently (As of July 10'th 2025), the newest version of python supported by 
sionna is 3.12. Installing python3.12:

| Ubuntu-based | Fedora | Windows |
| -- | -- | -- |
| ```sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.12``` | ```sudo dnf install python3.12```| Download it at [python.org](https://www.python.org/downloads/) |

- Setup a virtual environment, install the stuff, and run the stuff.

```bash
$ python3.12 -m venv env  # create the environment
$ source env/bin/activate  # activate it
$ pip install -r requirements.txt # install packages
$ ./env/bin/spyder # Run spyder or alternatively jupyterlab
$ ./env/bin/jupyterlab # Run spyder or alternatively jupyterlab
$ deactivate # deactivate it.
````
