from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

MOCK_DATA = {
    'item_map': {'A1': 'The Matrix', 'A2': 'Inception', 'A3': 'Interstellar'},
    'alias_map': {},
    'primary_names': {'A1': 'The Matrix', 'A2': 'Inception', 'A3': 'Interstellar'},
    'profiles': [
        {
            'user_id': 'USER1',
            'history': ['A1'],
            'might_like': ['A3'],
            'conversations': [
                {'conversation_id': 1, 'user_likes': ['A1'], 'user_dislikes': [], 'rec_item': []},
            ],
        },
    ],
    'conversations': {1: "User: Hello. Assistant: Hi there!"},
    'user_index': {'USER1': 0},
}


@pytest.fixture
def client():
    with patch('data_loader.load_item_map', return_value=MOCK_DATA['item_map']), \
         patch('data_loader.load_user_ids', return_value={}), \
         patch('data_loader.load_profiles', return_value=MOCK_DATA['profiles']), \
         patch('data_loader.load_conversations', return_value=MOCK_DATA['conversations']), \
         patch('data_loader.build_alias_map', return_value=({}, MOCK_DATA['primary_names'])), \
         patch('rag.crs.load_index', return_value=MagicMock()):

        import app as app_module
        app_module.data = MOCK_DATA

        with TestClient(app_module.app, raise_server_exceptions=False) as c:
            yield c


class TestHealthEndpoint:
    def test_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body['status'] == 'ok'
        assert body['movies'] == 3


class TestUsersEndpoint:
    def test_returns_users(self, client):
        resp = client.get("/users")
        assert resp.status_code == 200
        body = resp.json()
        assert 'USER1' in body['users']
        assert body['total'] == 1

    def test_respects_limit(self, client):
        resp = client.get("/users?limit=0")
        assert resp.json()['users'] == []

    def test_offset(self, client):
        resp = client.get("/users?offset=100")
        assert resp.json()['users'] == []


class TestGetUserEndpoint:
    def test_returns_user_info(self, client):
        resp = client.get("/users/USER1")
        assert resp.status_code == 200
        body = resp.json()
        assert body['user_id'] == 'USER1'
        assert body['history_count'] >= 1

    def test_404_for_unknown_user(self, client):
        resp = client.get("/users/NONEXISTENT")
        assert resp.status_code == 404


class TestMovieSearchEndpoint:
    def test_searches_movies(self, client):
        resp = client.get("/movies/search?q=matrix")
        assert resp.status_code == 200
        body = resp.json()
        titles = [r['title'] for r in body['results']]
        assert 'The Matrix' in titles

    def test_no_results(self, client):
        resp = client.get("/movies/search?q=zzzzzzzzz")
        assert resp.status_code == 200
        assert resp.json()['results'] == []


class TestRecommendEndpoint:
    def test_returns_stream(self, client):
        async def fake_stream(*args, **kwargs):
            yield "Hello "
            yield "World"

        with patch('app.few_shot_stream', side_effect=fake_stream):
            resp = client.post("/recommend", json={
                "user_id": "USER1",
                "message": "Recommend something!",
                "approach": "few_shot",
            })
            assert resp.status_code == 200
            assert "text/markdown" in resp.headers['content-type']

    def test_404_for_unknown_user(self, client):
        resp = client.post("/recommend", json={
            "user_id": "NOPE",
            "message": "hi",
        })
        assert resp.status_code == 404

    def test_default_approach_is_few_shot(self, client):
        async def fake_stream(*args, **kwargs):
            yield "ok"

        with patch('app.few_shot_stream', side_effect=fake_stream):
            resp = client.post("/recommend", json={
                "user_id": "USER1",
                "message": "hi",
            })
            assert resp.status_code == 200


class TestRecommendRequestValidation:
    def test_missing_user_id(self, client):
        resp = client.post("/recommend", json={"message": "hi"})
        assert resp.status_code == 422

    def test_missing_message(self, client):
        resp = client.post("/recommend", json={"user_id": "USER1"})
        assert resp.status_code == 422
