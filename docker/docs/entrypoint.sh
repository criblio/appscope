#!/bin/bash
set -e

echo "Starting ..."

echo "Resolving references in JSON Schema"

cd schemas

# This creates temporary schema.json files

for schema_file in *.schema.json; do
    [ -f "$schema_file" ] || break
    json-dereference -s "$schema_file" -o $TMP_DIR_NAME/"$schema_file"
done

echo "Finished resolving references in JSON Schema"

echo "Validating examples in JSON Schema"

python3 $EXAMPLE_TEST_PY

echo "Finished validating examples in JSON Schema"

echo "Generating Markdown files from JSON Schema"

node ../schema2md.js $TMP_DIR_NAME $MD_DIR_NAME

# Uncomment these lines to save the temporary schema.json files, which otherwise will be lost
# for resolve_schema_file in $TMP_DIR_NAME/*.schema.json; do
#     [ -f "$resolve_schema_file" ] || break
#     json-schema-gendoc $resolve_schema_file > $MD_DIR_NAME/${resolve_schema_file##*/}.md
# done

cp -rT $MD_DIR_NAME $MD_OUTPUT_PATH

echo "Finished generating Markdown files from JSON Schema"
