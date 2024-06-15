#!/bin/sh

rsync -avP \
  --exclude 'build' \
  --exclude 'build_3rdparty' \
  "$PWD"/. jeffrey@169.254.100.2:/projects/mirror-gaze


