#!/bin/bash
docker stop ghpages
docker run --rm -d -p 127.0.0.1:4000:4000 --name ghpages -it -v $(pwd):/src/site gh-pages /bin/bash
docker exec -it ghpages bash -l
