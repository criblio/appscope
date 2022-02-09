#!/usr/bin/env python3

import abc
import json
import jsonschema
import os
import sys

SCHEMA_RESOLVE_DIR = '/home/node/schemas/temp/'

class SchemaProperty(abc.ABC):

    def __init__(self, name:str, schema:dict) -> None:
        self.name = name
        self.schema = schema
        self.examples = schema.get('examples')

    def __str__(self) -> str:
        return self.name

    def validate_examples(self) -> None:
        if self.examples is None:
            print(f'Missing examples for {self}')
        else:
            for example in self.examples:
                jsonschema.validate(example, self.schema)

    @abc.abstractmethod
    def disable_unknown_property(self) -> None:
        pass

class Event(SchemaProperty):

    def disable_unknown_property(self) -> None:
        self.schema['properties']['body']['properties']['data'].update({'additionalProperties': False})


class Metric(SchemaProperty):

    def disable_unknown_property(self) -> None:
        self.schema['properties']['body'].update({'additionalProperties': False})


def is_event (schema:dict) -> bool:
    properties = schema.get('properties')
    type = properties.get('type')
    type_value = type.get('const')
    return type_value == 'evt'


def main():
    for schema_name in os.listdir(SCHEMA_RESOLVE_DIR):
        schema_path = os.path.join(SCHEMA_RESOLVE_DIR, schema_name)
        with open(schema_path, 'r') as f:
            schema_data = f.read()
        schema_json = json.loads(schema_data)

        schema = Event(schema_name, schema_json) if is_event(schema_json) else Metric(schema_name, schema_json)
        schema.disable_unknown_property()
        schema.validate_examples()


if __name__ == "__main__":
    sys.exit(main())