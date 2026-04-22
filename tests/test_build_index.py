from build_index import (
    MAX_DOC_CHARS,
    MAX_SNIPPETS,
    _build_movie_to_convos,
    _extract_snippets,
    build_enriched_docs,
)


def make_data():
    profiles = [
        {
            'user_id': 'U1',
            'history': ['A1', 'A2'],
            'might_like': [],
            'conversations': [
                {
                    'conversation_id': 10,
                    'user_likes': ['A1'],
                    'user_dislikes': ['A2'],
                    'rec_item': ['A3'],
                },
            ],
        },
        {
            'user_id': 'U2',
            'history': ['A3'],
            'might_like': [],
            'conversations': [
                {
                    'conversation_id': 20,
                    'user_likes': ['A3'],
                    'user_dislikes': [],
                    'rec_item': ['A1'],
                },
            ],
        },
    ]
    conversations = {
        10: "User: I love The Matrix, it is a great sci-fi film. Assistant: The Matrix is one of the best sci-fi films ever made.",
        20: "User: Interstellar was mind-blowing with the space scenes. Assistant: Yes, Interstellar explores space and time beautifully.",
    }
    return {
        'item_map': {'A1': 'The Matrix', 'A2': 'Inception', 'A3': 'Interstellar'},
        'alias_map': {},
        'primary_names': {'A1': 'The Matrix', 'A2': 'Inception', 'A3': 'Interstellar'},
        'profiles': profiles,
        'conversations': conversations,
        'user_index': {'U1': 0, 'U2': 1},
    }


# ── _build_movie_to_convos ──

class TestBuildMovieToConvos:
    def test_maps_likes(self):
        data = make_data()
        result = _build_movie_to_convos(data)
        assert 10 in result['A1']

    def test_maps_dislikes(self):
        data = make_data()
        result = _build_movie_to_convos(data)
        assert 10 in result['A2']

    def test_maps_rec_items(self):
        data = make_data()
        result = _build_movie_to_convos(data)
        assert 10 in result['A3']
        assert 20 in result['A3']

    def test_resolves_aliases(self):
        data = make_data()
        data['alias_map'] = {'A4': 'A1'}
        data['profiles'][0]['conversations'][0]['user_likes'].append('A4')
        result = _build_movie_to_convos(data)
        assert 10 in result['A1']

    def test_returns_sets(self):
        data = make_data()
        result = _build_movie_to_convos(data)
        for v in result.values():
            assert isinstance(v, set)


# ── _extract_snippets ──

class TestExtractSnippets:
    def test_extracts_relevant_sentences(self):
        text = "User: The Matrix is amazing. Assistant: I agree, The Matrix is a classic."
        snippets = _extract_snippets("The Matrix", text)
        assert len(snippets) > 0
        for s in snippets:
            assert "matrix" in s.lower()

    def test_strips_speaker_prefix(self):
        text = "User: The Matrix is truly groundbreaking film."
        snippets = _extract_snippets("The Matrix", text)
        assert not snippets[0].startswith("User:")

    def test_respects_max_snippets(self):
        text = ". ".join([f"Sentence about The Matrix number {i} is great" for i in range(20)])
        snippets = _extract_snippets("The Matrix", text)
        assert len(snippets) <= MAX_SNIPPETS

    def test_skips_short_sentences(self):
        text = "The Matrix. Also The Matrix is great film really."
        snippets = _extract_snippets("The Matrix", text)
        for s in snippets:
            assert len(s) > 15

    def test_case_insensitive(self):
        text = "I watched the matrix and it blew my mind completely."
        snippets = _extract_snippets("The Matrix", text)
        assert len(snippets) > 0

    def test_no_match_returns_empty(self):
        text = "I love watching sci-fi movies at night."
        snippets = _extract_snippets("The Matrix", text)
        assert snippets == []

    def test_deduplicates(self):
        text = ("User: The Matrix is a truly amazing film. "
                "Assistant: The Matrix is a truly amazing film.")
        snippets = _extract_snippets("The Matrix", text)
        assert len(snippets) == 1


# ── build_enriched_docs ──

class TestBuildEnrichedDocs:
    def test_includes_all_primary_movies(self):
        data = make_data()
        docs = build_enriched_docs(data)
        assert set(docs.keys()) == set(data['primary_names'].keys())

    def test_enriched_docs_have_snippets(self):
        data = make_data()
        docs = build_enriched_docs(data)
        enriched_count = sum(1 for d in docs.values() if '|' in d)
        assert enriched_count > 0

    def test_non_enriched_docs_are_title_only(self):
        data = make_data()
        data['profiles'] = []
        docs = build_enriched_docs(data)
        for asin, doc in docs.items():
            assert doc == data['primary_names'][asin]

    def test_doc_length_capped(self):
        data = make_data()
        docs = build_enriched_docs(data)
        for doc in docs.values():
            assert len(doc) <= MAX_DOC_CHARS
