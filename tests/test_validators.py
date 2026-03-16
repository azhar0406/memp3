import pytest

from memp3.core.validators import (
    ValidationError,
    validate_content,
    validate_memory_id,
    validate_query,
    validate_tags,
)


class TestValidateContent:
    def test_valid(self):
        assert validate_content("hello") == "hello"

    def test_empty_rejected(self):
        with pytest.raises(ValidationError, match="empty"):
            validate_content("")

    def test_whitespace_only_rejected(self):
        with pytest.raises(ValidationError, match="empty"):
            validate_content("   ")

    def test_oversized_rejected(self):
        with pytest.raises(ValidationError, match="exceeds"):
            validate_content("x" * 1_100_000)

    def test_non_string_rejected(self):
        with pytest.raises(ValidationError, match="string"):
            validate_content(123)


class TestValidateMemoryId:
    def test_valid_uuid(self):
        uid = "12345678-1234-1234-1234-123456789abc"
        assert validate_memory_id(uid) == uid

    def test_invalid_format(self):
        with pytest.raises(ValidationError, match="UUID"):
            validate_memory_id("not-a-uuid")

    def test_non_string(self):
        with pytest.raises(ValidationError, match="UUID"):
            validate_memory_id(42)


class TestValidateQuery:
    def test_valid(self):
        assert validate_query("search term") == "search term"

    def test_too_long(self):
        with pytest.raises(ValidationError, match="exceeds"):
            validate_query("x" * 1001)


class TestValidateTags:
    def test_none_ok(self):
        assert validate_tags(None) is None

    def test_valid(self):
        assert validate_tags("foo,bar") == "foo,bar"

    def test_too_long(self):
        with pytest.raises(ValidationError, match="exceed"):
            validate_tags("x" * 501)
