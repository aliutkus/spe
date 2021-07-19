#!/bin/bash

DATA_DIR=data

if [ ! -d "$DATA_DIR" ]
then
  mkdir "$DATA_DIR"
fi

if [ ! -f "$DATA_DIR"/lra_release.tar.gz ]
then
  wget https://storage.googleapis.com/long-range-arena/lra_release.gz
  mv lra_release.gz "$DATA_DIR"/lra_release.tar.gz
fi

tar -zxvf "$DATA_DIR"/lra_release.tar.gz lra_release/listops-1000 lra_release/lra_release/tsv_data
mv lra_release/lra_release/tsv_data lra_release
rm -rf lra_release/lra_release
mv lra_release "$DATA_DIR"