#!/bin/sh

rsync -avP --size-only "$PWD"/. jeffrey@169.254.100.2:/projects/mirror-gaze
