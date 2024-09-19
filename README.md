# Music generation Project

## Files and what they do
preprocessing.py : toolbox really, have functions for preprocessing, making dataset, and reconstructing the midifile for tuples generation style

working.py :  full transformers, generate by component, able to replicate input piece, no novelty

generate_by_pairs.py : full transformers, generate by tuple and reconstruct, able to replicate input piece, little to none novelty

## To run
Most of them are notebooks just click through :) if its python file you can run them like a regular python file

## Process

load midi -> [tokenize](https://miditok.readthedocs.io/en/latest/) -> generate dataset -> pass into model (full/decoder only transformers) -> generate new token/bar -> convert back into midi file -> use [this link](https://cifkao.github.io/html-midi-player/) to visualize

## Generation differences
By parts: generate components of the note separately (ex:pitch->bar position)

By tuples: generate a index of dictionary of tuples of seen notes (ex: (pitch, duration)) and recreate bar location using DPmapping in preprocessing.py (I should probably rename this) see more in [this report](https://docs.google.com/document/d/18mHfaBqByRVk7Hp4EF-RwjfDSWBy-mdExagcee3y8uo/edit?usp=sharing)