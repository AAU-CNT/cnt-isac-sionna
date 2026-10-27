#!/bin/bash

if ! test -d env; then
  echo "Venv does not exist. Creating."
  virtualenv env --python=python3.12
  if ! test -d env; then
    python3.12 -m venv env
    if ! test -d env; then
      python3 -m venv env
    fi
  fi
  source env/bin/activate
  pip install -r requirements_new_sionna.txt
else
  source env/bin/activate
fi

#
#if test -d .env; then
#  if ! test -d ../../sionna-rt; then
#    curdir=$(pwd)
#    cd ../../
#    echo "pulling sionna-rt fork"
#    git clone git@github.com:RobinJW/sionna-rt.git
#    cd $curdir
#  fi
#  pip install ../../sionna-rt
#fi
