"""Tests for src/serve/cutouts.py. The pure PNG-rendering logic is unit
tested directly; the thin Redis store/load wrapper is tested against the
real Redis container (infra/docker-compose.yml), same convention
test_serve_state.py already uses for src.serve.state's own Redis I/O.
"""

import numpy as np
import pytest
import redis
from PIL import Image

from src.serve import cutouts


def _fake_cutouts() -> np.ndarray:
    rng = np.random.default_rng(42)
    # A few bright pixels on an otherwise near-zero background -- the
    # exact ZScale use case (module docstring: linear scaling renders as
    # nothing on real astronomical cutouts).
    arr = rng.normal(loc=0.0, scale=1.0, size=(3, 21, 21)).astype(np.float32)
    arr[:, 10, 10] = 5000.0
    return arr


class TestPngRendering:
    def test_triplet_has_three_valid_pngs(self):
        images = cutouts.cutouts_to_png_triplet(_fake_cutouts())
        assert set(images.keys()) == set(cutouts.PLANE_NAMES)
        import base64
        import io

        for b64_png in images.values():
            img = Image.open(io.BytesIO(base64.b64decode(b64_png)))
            assert img.format == "PNG"
            assert img.size == (21, 21)

    def test_flat_plane_does_not_divide_by_zero(self):
        flat = np.zeros((3, 5, 5), dtype=np.float32)
        images = cutouts.cutouts_to_png_triplet(flat)
        assert len(images) == 3


@pytest.fixture()
def redis_client():
    client = redis.Redis(host="localhost", port=6379, decode_responses=True)
    try:
        client.ping()
    except redis.exceptions.ConnectionError:
        pytest.skip("Redis not reachable at localhost:6379 (start infra/docker-compose.yml)")
    yield client
    client.delete(cutouts.redis_key("__test_object__"))


class TestRedisRoundTrip:
    def test_store_and_load_round_trip(self, redis_client):
        arr = _fake_cutouts()
        cutouts.store_cutouts(redis_client, "__test_object__", arr)
        loaded = cutouts.load_cutouts(redis_client, "__test_object__")
        assert loaded is not None
        np.testing.assert_array_equal(loaded, arr)

    def test_missing_object_returns_none(self, redis_client):
        assert cutouts.load_cutouts(redis_client, "__object_never_stored__") is None
