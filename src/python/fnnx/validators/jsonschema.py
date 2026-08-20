import re


def validate_jsonschema(instance, schema, definitions=None):
    if definitions is None:
        definitions = {}
    schema = resolve_refs(schema, definitions)
    _validate(instance, schema, definitions)


def resolve_refs(schema, definitions, seen=None):
    if seen is None:
        seen = {}
    if id(schema) in seen:
        return seen[id(schema)]
    if isinstance(schema, dict):
        if "$defs" in schema:
            definitions.update(schema["$defs"])
        if "$ref" in schema:
            ref = schema["$ref"]
            if ref.startswith("#/$defs/"):
                def_name = ref.replace("#/$defs/", "")
                seen[id(schema)] = {}  # Placeholder
                resolved_schema = resolve_refs(definitions[def_name], definitions, seen)
                seen[id(schema)] = resolved_schema
                return resolved_schema
            else:
                raise ValueError(f"Unsupported $ref: {ref}")
        else:
            resolved_schema = {}
            seen[id(schema)] = resolved_schema  # Placeholder
            for k, v in schema.items():
                resolved_schema[k] = resolve_refs(v, definitions, seen)
            return resolved_schema
    elif isinstance(schema, list):
        return [resolve_refs(item, definitions, seen) for item in schema]
    else:
        return schema


def _validate(instance, schema, definitions):
    if "const" in schema:
        if instance != schema["const"]:
            raise ValueError(f'Value {instance} does not match const {schema["const"]}')
        return
    if "enum" in schema:
        if instance not in schema["enum"]:
            raise ValueError(f'Value {instance} is not in enum {schema["enum"]}')
        return
    if "not" in schema:
        try:
            _validate(instance, schema["not"], definitions)
        except ValueError:
            pass  # Valid because instance does not match 'not' schema
        else:
            raise ValueError('Instance should not match schema in "not"')
    if "anyOf" in schema:
        for subschema in schema["anyOf"]:
            try:
                _validate(instance, subschema, definitions)
                return
            except ValueError:
                continue
        raise ValueError(f"Value {instance} does not match any of the schemas in anyOf")
    if "allOf" in schema:
        for subschema in schema["allOf"]:
            _validate(instance, subschema, definitions)
        return
    if "oneOf" in schema:
        match_count = 0
        for subschema in schema["oneOf"]:
            try:
                _validate(instance, subschema, definitions)
                match_count += 1
            except ValueError:
                continue
        if match_count != 1:
            raise ValueError(
                f"Value {instance} must match exactly one schema in oneOf, but matched {match_count}"
            )
        return
    if "if" in schema:
        if_subschema = schema["if"]
        then_subschema = schema.get("then", {})
        else_subschema = schema.get("else", {})
        try:
            _validate(instance, if_subschema, definitions)
            _validate(instance, then_subschema, definitions)
        except ValueError:
            _validate(instance, else_subschema, definitions)
        return
    if "type" in schema:
        expected_types = schema["type"]
        if not isinstance(expected_types, list):
            expected_types = [expected_types]
        type_valid = False
        for expected_type in expected_types:
            if expected_type == "object" and isinstance(instance, dict):
                type_valid = True
            elif expected_type == "array" and isinstance(instance, list):
                type_valid = True
            elif expected_type == "string" and isinstance(instance, str):
                type_valid = True
            elif (
                expected_type == "integer"
                and isinstance(instance, int)
                and not isinstance(instance, bool)
            ):
                type_valid = True
            elif (
                expected_type == "number"
                and (isinstance(instance, int) or isinstance(instance, float))
                and not isinstance(instance, bool)
            ):
                type_valid = True
            elif expected_type == "boolean" and isinstance(instance, bool):
                type_valid = True
            elif expected_type == "null" and instance is None:
                type_valid = True
        if not type_valid:
            raise ValueError(
                f"Expected type {expected_types} but got {type(instance).__name__}"
            )
    if isinstance(instance, dict):
        _validate_object(instance, schema, definitions)
    elif isinstance(instance, list):
        _validate_array(instance, schema, definitions)
    elif isinstance(instance, str):
        _validate_string(instance, schema)
    elif isinstance(instance, (int, float)):
        _validate_number(instance, schema)


def _validate_object(instance, schema, definitions):
    if "required" in schema:
        for prop in schema["required"]:
            if prop not in instance:
                raise ValueError(f"Missing required property: {prop} in {instance}")
    total_properties = len(instance)
    if "minProperties" in schema:
        if total_properties < schema["minProperties"]:
            raise ValueError(
                f'Expected at least {schema["minProperties"]} properties, got {total_properties}'
            )
    if "maxProperties" in schema:
        if total_properties > schema["maxProperties"]:
            raise ValueError(
                f'Expected at most {schema["maxProperties"]} properties, got {total_properties}'
            )
    if "properties" in schema:
        for prop, subschema in schema["properties"].items():
            if prop in instance:
                _validate(instance[prop], subschema, definitions)
    if "patternProperties" in schema:
        for pattern, subschema in schema["patternProperties"].items():
            regex = re.compile(pattern)
            for prop in instance:
                if regex.match(prop):
                    _validate(instance[prop], subschema, definitions)
    if "additionalProperties" in schema:
        allowed_props = set(schema.get("properties", {}).keys())
        pattern_props: set[str] = set()
        if "patternProperties" in schema:
            for pattern in schema["patternProperties"]:
                for prop in instance:
                    if re.match(pattern, prop):
                        pattern_props.add(prop)
        extra_props = set(instance.keys()) - allowed_props - pattern_props

        if isinstance(schema["additionalProperties"], bool):
            if schema["additionalProperties"] is False and extra_props:
                raise ValueError(f"Additional properties not allowed: {extra_props}")
        else:
            for prop in extra_props:
                _validate(instance[prop], schema["additionalProperties"], definitions)
    elif "additionalProperties" not in schema and "properties" in schema:
        pass
    if "dependencies" in schema:
        for prop, dependency in schema["dependencies"].items():
            if prop in instance:
                if isinstance(dependency, list):
                    for dep_prop in dependency:
                        if dep_prop not in instance:
                            raise ValueError(
                                f"Property {prop} depends on {dep_prop}, which is missing"
                            )
                elif isinstance(dependency, dict):
                    _validate(instance, dependency, definitions)


def _validate_array(instance, schema, definitions):
    length = len(instance)
    if "minItems" in schema:
        if length < schema["minItems"]:
            raise ValueError(
                f'Array has {length} items, which is less than minItems {schema["minItems"]}'
            )
    if "maxItems" in schema:
        if length > schema["maxItems"]:
            raise ValueError(
                f'Array has {length} items, which is more than maxItems {schema["maxItems"]}'
            )
    if "uniqueItems" in schema and schema["uniqueItems"]:
        seen: set[object] = set()
        for item in instance:
            item_hash: object
            if isinstance(item, dict):
                item_hash = frozenset(item.items())
            elif isinstance(item, list):
                item_hash = tuple(item)
            else:
                item_hash = item
            if item_hash in seen:
                raise ValueError("Array items are not unique")
            seen.add(item_hash)
    if "items" in schema:
        items_schema = schema["items"]
        if isinstance(items_schema, dict):
            for item in instance:
                _validate(item, items_schema, definitions)
        elif isinstance(items_schema, list):
            for idx, item_schema in enumerate(items_schema):
                if idx < length:
                    _validate(instance[idx], item_schema, definitions)
                else:
                    break
            if "additionalItems" in schema:
                if not schema["additionalItems"]:
                    if length > len(items_schema):
                        raise ValueError("Additional items in array are not allowed")
                else:
                    additional_schema = schema["additionalItems"]
                    for idx in range(len(items_schema), length):
                        _validate(instance[idx], additional_schema, definitions)


def _validate_string(instance, schema):
    length = len(instance)
    if "minLength" in schema:
        if length < schema["minLength"]:
            raise ValueError(
                f'String length {length} is less than minLength {schema["minLength"]}'
            )
    if "maxLength" in schema:
        if length > schema["maxLength"]:
            raise ValueError(
                f'String length {length} is more than maxLength {schema["maxLength"]}'
            )
    if "pattern" in schema:
        if not re.match(schema["pattern"], instance):
            raise ValueError(f'String does not match pattern {schema["pattern"]}')
    if "format" in schema:
        format = schema["format"]
        if format == "email":
            # Simple email regex
            if not re.match(r"^[^@]+@[^@]+\.[^@]+$", instance):
                raise ValueError("String is not a valid email address")
        elif format == "uri":
            # Simple URI regex
            if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", instance):
                raise ValueError("String is not a valid URI")


def _validate_number(instance, schema):
    if "minimum" in schema:
        if instance < schema["minimum"]:
            raise ValueError(
                f'Value {instance} is less than minimum {schema["minimum"]}'
            )
    if "maximum" in schema:
        if instance > schema["maximum"]:
            raise ValueError(
                f'Value {instance} is greater than maximum {schema["maximum"]}'
            )
    if "exclusiveMinimum" in schema:
        if instance <= schema["exclusiveMinimum"]:
            raise ValueError(
                f'Value {instance} is less than or equal to exclusiveMinimum {schema["exclusiveMinimum"]}'
            )
    if "exclusiveMaximum" in schema:
        if instance >= schema["exclusiveMaximum"]:
            raise ValueError(
                f'Value {instance} is greater than or equal to exclusiveMaximum {schema["exclusiveMaximum"]}'
            )
    if "multipleOf" in schema:
        if (instance / schema["multipleOf"]) % 1 != 0:
            raise ValueError(
                f'Value {instance} is not a multiple of {schema["multipleOf"]}'
            )
