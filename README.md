# Music generation Project

## Files and what they do
working.py :  full transformers, able to replicate input piece

## To run
Most of them are notebooks just click through :) if its python file you can run them like a regular python file

## Process

load midi -> [tokenize](https://miditok.readthedocs.io/en/latest/) -> generate dataset -> pass into model (full/decoder only transformers) -> generate new token/bar -> convert back into midi file -> use [this link](https://cifkao.github.io/html-midi-player/) to visualize