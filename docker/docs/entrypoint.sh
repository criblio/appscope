#!/bin/bash
set -e

echo "Starting resolving references in JSON Schema"

cd schemas

for schema_file in *.schema.json; do
    [ -f "$schema_file" ] || break
    json-dereference -s "$schema_file" -o $TMP_DIR_NAME/"$schema_file"
done

echo "Starting generating MD files from JSON Schema"

for resolve_schema_file in $TMP_DIR_NAME/*.schema.json; do
    [ -f "$resolve_schema_file" ] || break
    json-schema-gendoc $resolve_schema_file > $MD_DIR_NAME/${resolve_schema_file##*/}.md
done

chmod 744 $MD_DIR_NAME
cp -r $MD_DIR_NAME $MD_OUTPUT_PATH

#Uncomment below to generate resolve JSON Schema

# chmod 744 $TMP_DIR_NAME
# cp -r $TMP_DIR_NAME $MD_OUTPUT_PATH

echo "Generation MD files from JSON Schema has finished"
