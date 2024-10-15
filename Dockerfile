FROM ghcr.io/conjure-cp/conjure:main

RUN apt-get update && apt-get upgrade -y && apt-get install -y python3 python3-pip python3-pandas python3-yaml python3-networkx python3-icecream python3-sklearn python3-numpy

RUN rm /root/.local/bin/savilerow.jar
RUN rm /root/.local/bin/cadical

COPY ./savilerow-main/savilerow.jar /root/.local/bin/
COPY ./savilerow-main/lib/json-20231013.jar /root/.local/bin/lib/
COPY ./cadical-learnt/bin/cadical-learnt /root/.local/bin/cadical
COPY ./NogoodParsing/nogoodparser /root/.local/bin/