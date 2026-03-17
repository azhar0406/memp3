import numpy as np
import pytest

from memp3.core.multichannel import MultiChannelEncoder


class TestMultiChannelEncoder:
    def test_ascii_round_trip(self):
        enc = MultiChannelEncoder()
        text = "Hello, World! 12345"
        assert enc.decode(enc.encode(text)) == text

    def test_unicode_round_trip(self):
        enc = MultiChannelEncoder()
        text = "Meeting at 3pm in Thane"
        assert enc.decode(enc.encode(text)) == text

    def test_large_text(self):
        enc = MultiChannelEncoder()
        text = "The quick brown fox. " * 500
        assert enc.decode(enc.encode(text)) == text

    def test_corruption_recovery(self):
        enc = MultiChannelEncoder()
        text = "Error correction test with multi-channel encoding." * 3
        signal = enc.encode(text)

        rng = np.random.default_rng(42)
        n = int(len(signal) * 0.03)
        indices = rng.choice(len(signal), n, replace=False)
        signal[indices] += rng.uniform(-0.2, 0.2, n)

        assert enc.decode(signal) == text

    def test_smaller_signal_than_single_channel(self):
        from memp3.core.encoder import BinaryEncoder
        mc = MultiChannelEncoder()
        sc = BinaryEncoder()
        text = "x" * 1000
        assert len(mc.encode(text)) < len(sc.encode(text))

    def test_truncated_raises(self):
        enc = MultiChannelEncoder()
        signal = enc.encode("test")
        with pytest.raises(ValueError):
            enc.decode(signal[: len(signal) // 4])
