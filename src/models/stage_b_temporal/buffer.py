"""
Reference buffer: recursive, never re-anchored to G_0 (design_doc_v2.md Section 0 / 2.1).

    buffer.read()       -> returns G_{t-1} (or G_0 if this is the first call after seeding)
    buffer.write(G_t)    -> overwrites buffer with G_t (the deformed output), for next frame

TODO(Phase 4):
    - implement as a simple stateful object first (dict of tensors), no persistence needed
      beyond one training clip / inference sequence
    - unit test: assert buffer.read() after one write() is NOT identical to the original
      seed G_0 (catches an accidental silent re-anchoring bug)
"""


class ReferenceBuffer:
    def __init__(self, g0):
        self._current = g0

    def read(self):
        return self._current

    def write(self, g_t):
        self._current = g_t
