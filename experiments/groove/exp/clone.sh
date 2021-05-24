#!/bin/bash
set -ex
mkdir "$2"
cp "$1/config.yaml" "$2/config.yaml"
sensible-editor "$2/config.yaml"
