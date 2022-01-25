FROM node

RUN npm install -g json-dereference-cli json-schema-gendoc

USER node
WORKDIR /home/node/

ENV MD_DIR_NAME="md_files"
ENV MD_OUTPUT_PATH="/md"
ENV TMP_DIR_NAME="temp"

COPY docker/docs/entrypoint.sh /home/node/entrypoint.sh

RUN mkdir -p /home/node/schemas/

COPY docs/schemas /home/node/schemas

RUN mkdir /home/node/schemas/temp
RUN mkdir /home/node/schemas/md_files

ENTRYPOINT ["/home/node//entrypoint.sh"]
