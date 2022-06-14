FROM node:17.9-slim

RUN npm install -g json-dereference-cli json-schema-gendoc
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN pip3 install jsonschema

USER root
WORKDIR /home/node/

ENV MD_DIR_NAME="md_files"
ENV MD_OUTPUT_PATH="/home/node/appscope/website/src/pages/docs/"
ENV TMP_DIR_NAME="temp"
ENV EXAMPLE_TEST_PY="/home/node/validation/main.py"

COPY docker/docs/entrypoint.sh /home/node/entrypoint.sh
COPY docker/docs/schema2md.js /home/node/schema2md.js
COPY docker/docs/layout.js /home/node/layout.js
COPY docker/docs/validation /home/node/validation

RUN mkdir -p /home/node/schemas/

COPY docs/schemas /home/node/schemas

RUN mkdir /home/node/schemas/temp
RUN mkdir /home/node/schemas/md_files

ENTRYPOINT ["/home/node/entrypoint.sh"]
