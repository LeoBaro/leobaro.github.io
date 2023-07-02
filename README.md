# LeoBaro.github.io

This website is built using the Docker image of https://github.com/github/pages-gem


cd <project-dir>
docker run --rm -d -p 127.0.0.1:4000:4000 --name ghpages -it -v $(pwd):/src/site gh-pages /bin/bash
docker exec -it ghpages bash -l
bundle install
bundle exec jekyll build 
bundle exec jekyll serve -H 0.0.0.0 -P 4000 --watch