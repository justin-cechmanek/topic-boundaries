"""Check REDIS_URL synthesis from Redis Cloud's discrete fields."""

from tests.conftest import redis_url_from_parts


def test_builds_authed_url():
    url = redis_url_from_parts(
        {
            "REDIS_HOST": "redis-13579.example.cloud.rlrcp.com",
            "REDIS_PORT": "13579",
            "REDIS_USERNAME": "default",
            "REDIS_PASSWORD": "secret",
        }
    )
    assert url == "redis://default:secret@redis-13579.example.cloud.rlrcp.com:13579"


def test_no_host_returns_none():
    assert redis_url_from_parts({}) is None


def test_no_password_omits_auth():
    url = redis_url_from_parts({"REDIS_HOST": "localhost", "REDIS_PORT": "6379"})
    assert url == "redis://localhost:6379"
