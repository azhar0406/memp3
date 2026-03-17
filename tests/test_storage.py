import pytest

from memp3.core.validators import ValidationError


class TestStorageManager:
    def test_store_and_retrieve(self, storage):
        mem_id = storage.store("Hello, memp3!")
        content = storage.retrieve(mem_id)
        assert content == "Hello, memp3!"

    def test_store_with_tags(self, storage):
        mem_id = storage.store("tagged memory", tags="test,demo")
        info = storage.get_info(mem_id)
        assert info["tags"] == "test,demo"

    def test_retrieve_not_found(self, storage):
        with pytest.raises(KeyError):
            storage.retrieve("00000000-0000-0000-0000-000000000000")

    def test_search(self, storage):
        storage.store("alpha beta gamma")
        storage.store("delta epsilon")
        results = storage.search("beta")
        assert len(results) == 1
        assert "beta" in results[0]["content"]

    def test_list_all(self, storage):
        storage.store("first")
        storage.store("second")
        results = storage.list_all()
        assert len(results) == 2

    def test_delete(self, storage):
        mem_id = storage.store("to be deleted")
        storage.delete(mem_id)
        with pytest.raises(KeyError):
            storage.retrieve(mem_id)

    def test_delete_not_found(self, storage):
        result = storage.delete("00000000-0000-0000-0000-000000000000")
        assert result is False

    def test_get_info(self, storage):
        mem_id = storage.store("info test")
        info = storage.get_info(mem_id)
        assert info["id"] == mem_id
        assert info["content_length"] == len("info test")
        assert info["encoder_version"] == 2
        assert info["flac_bytes"] > 0
        assert info["storage"] == "flac_blob"

    def test_stats(self, storage):
        storage.store("one")
        storage.store("two")
        s = storage.stats()
        assert s["total_memories"] == 2
        assert s["total_content_bytes"] > 0
        assert s["total_flac_bytes"] > 0

    def test_validation_bad_id(self, storage):
        with pytest.raises(ValidationError):
            storage.retrieve("bad-id")

    def test_validation_oversized_content(self, storage):
        with pytest.raises(ValidationError):
            storage.store("x" * 1_100_000)
