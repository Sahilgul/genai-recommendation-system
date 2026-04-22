from unittest.mock import MagicMock, patch

import rag.crs as rag_crs_module
from rag.crs import LMStudioEmbedding, retrieve


def make_profile():
    return {
        'user_id': 'U1',
        'history': ['A1', 'A2', 'A3'],
        'might_like': [],
        'conversations': [],
    }


def make_data():
    return {
        'item_map': {'A1': 'Matrix', 'A2': 'Inception', 'A3': 'Interstellar'},
        'alias_map': {},
        'primary_names': {'A1': 'Matrix', 'A2': 'Inception', 'A3': 'Interstellar'},
    }


class TestRetrieve:
    def test_filters_history(self):
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            'ids': [['A1', 'A4', 'A5']],
            'documents': [['Matrix', 'Blade Runner', 'Alien']],
            'distances': [[0.1, 0.2, 0.3]],
        }

        original = rag_crs_module.collection
        rag_crs_module.collection = mock_collection
        try:
            profile = make_profile()
            data = make_data()
            results = retrieve("sci-fi", profile, data, n_results=5)
            asins = [r['asin'] for r in results]
            assert 'A1' not in asins
            assert 'A4' in asins
        finally:
            rag_crs_module.collection = original

    def test_returns_correct_structure(self):
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            'ids': [['A4']],
            'documents': [['Blade Runner']],
            'distances': [[0.2]],
        }

        original = rag_crs_module.collection
        rag_crs_module.collection = mock_collection
        try:
            profile = make_profile()
            data = make_data()
            results = retrieve("sci-fi", profile, data)
            assert len(results) == 1
            assert results[0]['asin'] == 'A4'
            assert results[0]['title'] == 'Blade Runner'
            assert 'score' in results[0]
            assert results[0]['score'] == round(1 - 0.2, 4)
        finally:
            rag_crs_module.collection = original

    def test_respects_n_results(self):
        ids = [f'X{i}' for i in range(20)]
        docs = [f'Movie {i}' for i in range(20)]
        dists = [0.1 * i for i in range(20)]

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            'ids': [ids],
            'documents': [docs],
            'distances': [dists],
        }

        original = rag_crs_module.collection
        rag_crs_module.collection = mock_collection
        try:
            profile = {'user_id': 'U1', 'history': [], 'might_like': [], 'conversations': []}
            data = {'alias_map': {}}
            results = retrieve("test", profile, data, n_results=5)
            assert len(results) == 5
        finally:
            rag_crs_module.collection = original


class TestLMStudioEmbedding:
    def test_callable(self):
        embed_fn = LMStudioEmbedding()
        assert callable(embed_fn)

    @patch('rag.crs.embed_sync', return_value=[[0.1, 0.2, 0.3]])
    def test_calls_embed_sync(self, mock_embed):
        embed_fn = LMStudioEmbedding()
        result = embed_fn(["test"])
        mock_embed.assert_called_once_with(["test"])
        assert len(result) == 1
        assert len(result[0]) == 3
