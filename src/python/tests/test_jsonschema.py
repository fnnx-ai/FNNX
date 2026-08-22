import unittest
from fnnx.validators.jsonschema import validate_jsonschema


class TestValidateJSONSchema(unittest.TestCase):
    def test_const_valid(self):
        schema = {"const": 5}
        instance = 5
        validate_jsonschema(instance, schema)

    def test_const_invalid(self):
        schema = {"const": 5}
        instance = 6
        with self.assertRaises(ValueError):
            validate_jsonschema(instance, schema)

    def test_enum_valid(self):
        schema = {"enum": [1, 2, 3]}
        instance = 2
        validate_jsonschema(instance, schema)

    def test_enum_invalid(self):
        schema = {"enum": [1, 2, 3]}
        instance = 4
        with self.assertRaises(ValueError):
            validate_jsonschema(instance, schema)

    def test_type_string(self):
        schema = {"type": "string"}
        instance = "test"
        validate_jsonschema(instance, schema)

    def test_type_string_invalid(self):
        schema = {"type": "string"}
        instance = 5
        with self.assertRaises(ValueError):
            validate_jsonschema(instance, schema)

    def test_type_array(self):
        schema = {"type": "array"}
        instance = [1, 2, 3]
        validate_jsonschema(instance, schema)

    def test_type_array_invalid(self):
        schema = {"type": "array"}
        instance = "not an array"
        with self.assertRaises(ValueError):
            validate_jsonschema(instance, schema)

    def test_minItems(self):
        schema = {"type": "array", "minItems": 2}
        instance = [1, 2]
        validate_jsonschema(instance, schema)

    def test_minItems_invalid(self):
        schema = {"type": "array", "minItems": 3}
        instance = [1, 2]
        with self.assertRaises(ValueError):
            validate_jsonschema(instance, schema)

    def test_maxItems(self):
        schema = {"type": "array", "maxItems": 2}
        instance = [1, 2]
        validate_jsonschema(instance, schema)

    def test_maxItems_invalid(self):
        schema = {"type": "array", "maxItems": 1}
        instance = [1, 2]
        with self.assertRaises(ValueError):
            validate_jsonschema(instance, schema)

    def test_uniqueItems_valid(self):
        schema = {"type": "array", "uniqueItems": True}
        instance = [1, 2, 3]
        validate_jsonschema(instance, schema)

    def test_uniqueItems_invalid(self):
        schema = {"type": "array", "uniqueItems": True}
        instance = [1, 2, 2]
        with self.assertRaises(ValueError):
            validate_jsonschema(instance, schema)

    def test_minLength(self):
        schema = {"type": "string", "minLength": 3}
        instance = "abc"
        validate_jsonschema(instance, schema)

    def test_minLength_invalid(self):
        schema = {"type": "string", "minLength": 4}
        instance = "abc"
        with self.assertRaises(ValueError):
            validate_jsonschema(instance, schema)

    def test_maxLength(self):
        schema = {"type": "string", "maxLength": 3}
        instance = "abc"
        validate_jsonschema(instance, schema)

    def test_maxLength_invalid(self):
        schema = {"type": "string", "maxLength": 2}
        instance = "abc"
        with self.assertRaises(ValueError):
            validate_jsonschema(instance, schema)

    def test_pattern_valid(self):
        schema = {"type": "string", "pattern": "^a.*b$"}
        instance = "a test b"
        validate_jsonschema(instance, schema)

    def test_pattern_invalid(self):
        schema = {"type": "string", "pattern": "^a.*b$"}
        instance = "no match"
        with self.assertRaises(ValueError):
            validate_jsonschema(instance, schema)

    def test_number_minimum(self):
        schema = {"type": "number", "minimum": 5}
        instance = 5
        validate_jsonschema(instance, schema)

    def test_number_minimum_invalid(self):
        schema = {"type": "number", "minimum": 5}
        instance = 4
        with self.assertRaises(ValueError):
            validate_jsonschema(instance, schema)

    def test_number_exclusiveMinimum(self):
        schema = {"type": "number", "exclusiveMinimum": 5}
        instance = 6
        validate_jsonschema(instance, schema)

    def test_number_exclusiveMinimum_invalid(self):
        schema = {"type": "number", "exclusiveMinimum": 5}
        instance = 5
        with self.assertRaises(ValueError):
            validate_jsonschema(instance, schema)

    def test_multipleOf_valid(self):
        schema = {"type": "number", "multipleOf": 2}
        instance = 4
        validate_jsonschema(instance, schema)

    def test_multipleOf_invalid(self):
        schema = {"type": "number", "multipleOf": 2}
        instance = 5
        with self.assertRaises(ValueError):
            validate_jsonschema(instance, schema)

    def test_required_properties(self):
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
        }
        instance = {"name": "John", "age": 30}
        validate_jsonschema(instance, schema)

    def test_required_properties_missing(self):
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
        }
        instance = {"name": "John"}
        with self.assertRaises(ValueError):
            validate_jsonschema(instance, schema)

    def test_additionalProperties_false(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": False,
        }
        instance = {"name": "John"}
        validate_jsonschema(instance, schema)

    def test_additionalProperties_false_invalid(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": False,
        }
        instance = {"name": "John", "age": 30}
        with self.assertRaises(ValueError):
            validate_jsonschema(instance, schema)

    def test_patternProperties(self):
        schema = {
            "type": "object",
            "patternProperties": {"^S_": {"type": "string"}},
        }
        instance = {"S_name": "John", "S_age": "30"}
        validate_jsonschema(instance, schema)

    def test_patternProperties_invalid(self):
        schema = {
            "type": "object",
            "patternProperties": {"^S_": {"type": "string"}},
        }
        instance = {"S_name": "John", "S_age": 30}
        with self.assertRaises(ValueError):
            validate_jsonschema(instance, schema)

    def test_dependencies_property(self):
        schema = {
            "type": "object",
            "properties": {"credit_card": {"type": "string"}},
            "dependencies": {"credit_card": ["billing_address"]},
        }
        instance = {
            "credit_card": "1234-5678-9012-3456",
            "billing_address": "123 Street",
        }
        validate_jsonschema(instance, schema)

    def test_dependencies_property_missing(self):  ### This test is failing
        schema = {
            "type": "object",
            "properties": {"credit_card": {"type": "string"}},
            "dependencies": {"credit_card": ["billing_address"]},
        }
        instance = {"credit_card": "1234-5678-9012-3456"}
        with self.assertRaises(ValueError):
            validate_jsonschema(instance, schema)

    def test_ref_definition(self):
        schema = {
            "$defs": {"positiveInteger": {"type": "integer", "minimum": 0}},
            "type": "object",
            "properties": {"age": {"$ref": "#/$defs/positiveInteger"}},
        }
        instance = {"age": 30}
        validate_jsonschema(instance, schema)

    def test_ref_definition_invalid(self):
        schema = {
            "$defs": {"positiveInteger": {"type": "integer", "minimum": 0}},
            "type": "object",
            "properties": {"age": {"$ref": "#/$defs/positiveInteger"}},
        }
        instance = {"age": -5}
        with self.assertRaises(ValueError):
            validate_jsonschema(instance, schema)

    def test_allOf(self):
        schema = {"allOf": [{"type": "integer"}, {"minimum": 2}]}
        instance = 5
        validate_jsonschema(instance, schema)

    def test_allOf_invalid(self):
        schema = {"allOf": [{"type": "integer"}, {"minimum": 2}]}
        instance = 1
        with self.assertRaises(ValueError):
            validate_jsonschema(instance, schema)

    def test_anyOf(self):
        schema = {"anyOf": [{"type": "string"}, {"type": "number"}]}
        validate_jsonschema("test", schema)
        validate_jsonschema(5, schema)

    def test_anyOf_invalid(self):
        schema = {"anyOf": [{"type": "string"}, {"type": "number"}]}
        instance = True
        with self.assertRaises(ValueError):
            validate_jsonschema(instance, schema)

    def test_oneOf(self):
        schema = {
            "oneOf": [
                {"type": "string", "maxLength": 5},
                {"type": "string", "minLength": 9},
            ]
        }
        instance = "abcab"
        validate_jsonschema(instance, schema)

        instance = "abcabcabc"
        validate_jsonschema(instance, schema)

    def test_oneOf_invalid(self):
        schema = {
            "oneOf": [
                {"type": "string", "minLength": 2},
                {"type": "string", "maxLength": 5},
            ]
        }
        instance = "abcd"
        with self.assertRaises(ValueError):
            validate_jsonschema(instance, schema)

    def test_not(self):
        schema = {"not": {"type": "string"}}
        instance = 5
        validate_jsonschema(instance, schema)

    def test_not_invalid(self):
        schema = {"not": {"type": "string"}}
        instance = "test"
        with self.assertRaises(ValueError):
            validate_jsonschema(instance, schema)

    def test_if_then_else_then(self):
        schema = {
            "if": {"type": "number"},
            "then": {"minimum": 10},
            "else": {"type": "string"},
        }
        instance = 15
        validate_jsonschema(instance, schema)

    def test_if_then_else_else(self):
        schema = {
            "if": {"type": "number"},
            "then": {"minimum": 10},
            "else": {"type": "string"},
        }
        instance = "test"
        validate_jsonschema(instance, schema)

    def test_if_then_else_then_invalid(self):
        schema = {
            "if": {"type": "number"},
            "then": {"minimum": 10},
            "else": {"type": "string"},
        }
        instance = 5
        with self.assertRaises(ValueError):
            validate_jsonschema(instance, schema)

    def test_if_then_else_else_invalid(self):
        schema = {
            "if": {"type": "number"},
            "then": {"minimum": 10},
            "else": {"type": "string"},
        }
        instance = True
        with self.assertRaises(ValueError):
            validate_jsonschema(instance, schema)

    def test_null_type(self):
        schema = {"type": "null"}
        instance = None
        validate_jsonschema(instance, schema)

    def test_null_type_invalid(self):
        schema = {"type": "null"}
        instance = "not null"
        with self.assertRaises(ValueError):
            validate_jsonschema(instance, schema)

    def test_boolean_type(self):
        schema = {"type": "boolean"}
        instance = True
        validate_jsonschema(instance, schema)

    def test_boolean_type_invalid(self):
        schema = {"type": "boolean"}
        instance = "True"
        with self.assertRaises(ValueError):
            validate_jsonschema(instance, schema)

    def test_format_email_valid(self):
        schema = {"type": "string", "format": "email"}
        instance = "test@example.com"
        validate_jsonschema(instance, schema)

    def test_format_email_invalid(self):
        schema = {"type": "string", "format": "email"}
        instance = "invalid-email"
        with self.assertRaises(ValueError):
            validate_jsonschema(instance, schema)

    def test_format_uri_valid(self):
        schema = {"type": "string", "format": "uri"}
        instance = "http://example.com"
        validate_jsonschema(instance, schema)

    def test_format_uri_invalid(self):
        schema = {"type": "string", "format": "uri"}
        instance = "not a uri"
        with self.assertRaises(ValueError):
            validate_jsonschema(instance, schema)

    def test_multiple_types_valid(self):
        schema = {"type": ["string", "number"]}
        validate_jsonschema("test", schema)
        validate_jsonschema(5, schema)

    def test_multiple_types_invalid(self):
        schema = {"type": ["string", "number"]}
        instance = True
        with self.assertRaises(ValueError):
            validate_jsonschema(instance, schema)

    def test_unknown_keyword(self):
        schema = {"unknownKeyword": True}
        instance = {"test": "value"}
        validate_jsonschema(instance, schema)  # Should ignore unknown keywords

    def test_recursive_schema(self):
        schema = {
            "$defs": {
                "node": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number"},
                        "next": {"$ref": "#/$defs/node"},
                    },
                    "required": ["value"],
                }
            },
            "$ref": "#/$defs/node",
        }
        instance = {"value": 1, "next": {"value": 2, "next": {"value": 3}}}
        validate_jsonschema(instance, schema)

    def test_recursive_schema_invalid(self):
        schema = {
            "$defs": {
                "node": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number"},
                        "next": {"$ref": "#/$defs/node"},
                    },
                    "required": ["value"],
                }
            },
            "$ref": "#/$defs/node",
        }
        instance = {"value": 1, "next": {"value": "not a number"}}
        with self.assertRaises(ValueError):
            validate_jsonschema(instance, schema)

    def test_additionalItems_false(self):
        schema = {
            "type": "array",
            "items": [{"type": "number"}, {"type": "string"}],
            "additionalItems": False,
        }
        instance = [1, "test"]
        validate_jsonschema(instance, schema)

    def test_additionalItems_false_invalid(self):
        schema = {
            "type": "array",
            "items": [{"type": "number"}, {"type": "string"}],
            "additionalItems": False,
        }
        instance = [1, "test", True]
        with self.assertRaises(ValueError):
            validate_jsonschema(instance, schema)

    def test_additionalItems_schema(self):
        schema = {
            "type": "array",
            "items": [{"type": "number"}],
            "additionalItems": {"type": "string"},
        }
        instance = [1, "test", "another"]
        validate_jsonschema(instance, schema)

    def test_additionalItems_schema_invalid(self):
        schema = {
            "type": "array",
            "items": [{"type": "number"}],
            "additionalItems": {"type": "string"},
        }
        instance = [1, "test", 3]
        with self.assertRaises(ValueError):
            validate_jsonschema(instance, schema)

    def test_empty_schema(self):
        schema: dict[str, object] = {}
        instance = "anything"
        validate_jsonschema(instance, schema)  # Should accept any instance

    def test_escape_characters_in_pattern(self):
        schema = {"type": "string", "pattern": "^\\d{3}-\\d{2}-\\d{4}$"}
        instance = "123-45-6789"
        validate_jsonschema(instance, schema)

    def test_escape_characters_in_pattern_invalid(self):
        schema = {"type": "string", "pattern": "^\\d{3}-\\d{2}-\\d{4}$"}
        instance = "abc-de-ghij"
        with self.assertRaises(ValueError):
            validate_jsonschema(instance, schema)


class TestSiblingKeywords(unittest.TestCase):
    """Every in-subset keyword of a schema object is enforced, not just the first one."""

    def test_const_does_not_shadow_a_sibling(self):
        schema = {"const": {"a": 1}, "type": "array"}
        with self.assertRaises(ValueError):
            validate_jsonschema({"a": 1}, schema)

    def test_enum_does_not_shadow_a_sibling(self):
        schema = {"enum": [{"a": 1}], "required": ["b"]}
        with self.assertRaises(ValueError):
            validate_jsonschema({"a": 1}, schema)

    def test_any_of_does_not_shadow_a_sibling(self):
        schema = {
            "anyOf": [{"type": "object"}, {"type": "array"}],
            "required": ["b"],
        }
        validate_jsonschema({"b": 1}, schema)
        with self.assertRaises(ValueError):
            validate_jsonschema({"a": 1}, schema)

    def test_all_of_does_not_shadow_a_sibling(self):
        schema = {"allOf": [{"type": "object"}], "required": ["b"]}
        with self.assertRaises(ValueError):
            validate_jsonschema({"a": 1}, schema)

    def test_one_of_does_not_shadow_a_sibling(self):
        schema = {
            "oneOf": [{"type": "object"}, {"type": "array"}],
            "required": ["b"],
        }
        with self.assertRaises(ValueError):
            validate_jsonschema({"a": 1}, schema)

    def test_if_then_else_does_not_shadow_a_sibling(self):
        schema = {
            "if": {"required": ["a"]},
            "then": {"required": ["b"]},
            "else": {"required": ["c"]},
            "type": "object",
            "maxProperties": 2,
        }
        validate_jsonschema({"a": 1, "b": 2}, schema)
        with self.assertRaises(ValueError):
            validate_jsonschema([1], schema)

    def test_then_failure_does_not_fall_back_to_else(self):
        schema = {
            "if": {"required": ["a"]},
            "then": {"required": ["b"]},
            "else": {"required": ["a"]},
        }
        with self.assertRaises(ValueError):
            validate_jsonschema({"a": 1}, schema)


class TestUnanchoredPatterns(unittest.TestCase):
    """`pattern` and `patternProperties` are searched, not anchored at the start."""

    def test_pattern_matches_anywhere_in_the_string(self):
        validate_jsonschema("abc-123-def", {"type": "string", "pattern": "123"})

    def test_pattern_anchored_at_the_end_is_enforced(self):
        with self.assertRaises(ValueError):
            validate_jsonschema("123-abc", {"type": "string", "pattern": "\\d+$"})

    def test_pattern_properties_match_anywhere_in_the_key(self):
        schema = {
            "type": "object",
            "patternProperties": {"count$": {"type": "integer"}},
            "additionalProperties": False,
        }
        validate_jsonschema({"item_count": 2}, schema)
        with self.assertRaises(ValueError):
            validate_jsonschema({"item_count": "two"}, schema)


if __name__ == "__main__":
    unittest.main()
