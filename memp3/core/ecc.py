"""Error correction for memp3 audio data."""

from reedsolo import RSCodec


class ReedSolomonECC:
    """Reed-Solomon error correction wrapper.

    Uses RS(255, 223) — 32 ECC symbols per 223 data bytes (~14% overhead).
    Data is processed in chunks to handle arbitrarily large payloads.
    """

    def __init__(self, nsym: int = 32):
        self.nsym = nsym
        self._codec = RSCodec(nsym)

    def encode(self, data: bytes) -> bytes:
        """Add Reed-Solomon error correction codes to data."""
        return bytes(self._codec.encode(data))

    def decode(self, data: bytes) -> bytes:
        """Decode and correct errors in RS-encoded data."""
        decoded = self._codec.decode(data)
        # reedsolo.decode returns (data_bytearray, remainder_bytearray, errata_pos)
        return bytes(decoded[0])
