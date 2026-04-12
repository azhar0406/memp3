import numpy as np
import pytest

from memdio.core.encoder import BinaryEncoder, SimpleEncoder


class TestSimpleEncoder:
    def test_round_trip(self):
        enc = SimpleEncoder()
        text = "Hello World"
        signal = enc.encode(text)
        decoded = enc.decode(signal)
        assert decoded == text

    def test_unknown_char_fallback(self):
        enc = SimpleEncoder()
        freq = enc.char_to_freq("\x00")
        assert enc.min_freq <= freq <= enc.max_freq


class TestBinaryEncoder:
    def test_ascii_round_trip(self):
        enc = BinaryEncoder()
        text = "Hello, World! 12345"
        assert enc.decode(enc.encode(text)) == text

    def test_unicode_round_trip(self):
        enc = BinaryEncoder()
        text = "こんにちは 🌍 café"
        assert enc.decode(enc.encode(text)) == text

    def test_empty_string_rejected(self):
        """Empty string should fail at validation level, not encoder."""
        enc = BinaryEncoder()
        # Encoder itself handles empty — it just compresses empty bytes
        sig = enc.encode("")
        assert enc.decode(sig) == ""

    def test_large_text(self):
        enc = BinaryEncoder()
        text = "The quick brown fox. " * 500  # ~10KB
        assert enc.decode(enc.encode(text)) == text

    def test_corruption_recovery(self):
        """5% sample corruption should be recoverable via RS ECC."""
        enc = BinaryEncoder()
        text = "Error correction test with enough data for RS to work properly." * 3
        signal = enc.encode(text)

        rng = np.random.default_rng(42)
        n_corrupt = int(len(signal) * 0.05)
        indices = rng.choice(len(signal), n_corrupt, replace=False)
        signal[indices] += rng.uniform(-0.3, 0.3, n_corrupt)

        assert enc.decode(signal) == text

    def test_truncated_data_raises(self):
        enc = BinaryEncoder()
        signal = enc.encode("test")
        # Truncate heavily
        with pytest.raises(ValueError):
            enc.decode(signal[: len(signal) // 4])
