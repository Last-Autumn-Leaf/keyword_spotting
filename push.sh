#!/bin/bash
git add \*.ipynb \*.png \*.pt \*.out \*.py \*.sh \*txt \*bin

git add runs\*
git add *.ipynb
git add *.png
git add *.pt
git add *.py
git add *.out
git add *.sh
git commit -m 'Modified version'
git push