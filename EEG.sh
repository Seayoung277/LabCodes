#!/bin/sh
for ((i=1; i<16; i++))
do
	python EEG_MLP_TF.py -l 256 256 256 64 -n 0 -o 1 -b 1000 -e 1000 -s 1 -c $i -d 0.9 -a 1
done
