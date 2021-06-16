#!/bin/bash

cd data 
mkdir -p genome/mm10 && cd genome/mm10
wget http://hgdownload.cse.ucsc.edu/goldenPath/mm10/bigZips/chromFa.tar.gz
tar zvfx chromFa.tar.gz
cd ..
mkdir hg19 && cd hg19
wget http://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/chromFa.tar.gz
tar zvfx chromFa.tar.gz 
cd ../../..