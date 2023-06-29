#!/bin/bash

set -e 

bundle install
bundle exec jekyll build 
bundle exec jekyll serve -H 0.0.0.0 -P 4000 --watch