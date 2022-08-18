#!/bin/bash
#git add \*.ipynb \*.png \*.pt \*.out \*.py \*.sh \*.txt \*.bin
git add runs/* *.sh *.out *.py *.pt *.png *.ipynb *.txt *.bin
git add ./logs
git add ./runs
git commit -m 'Modified version'
git push