import pytest
from data_loader import (
    normalize_title, clean_name, build_alias_map, resolve,
    get_movie_name, get_user_history_names, search_movies, load_all,
)


# ── normalize_title ──

class TestNormalizeTitle:
    def test_strips_vhs_tag(self):
        assert "white christmas" == normalize_title("White Christmas VHS")

    def test_strips_dvd_tag(self):
        assert "white christmas" == normalize_title("White Christmas DVD")

    def test_strips_bluray_tag(self):
        assert "white christmas" == normalize_title("White Christmas Blu-ray")

    def test_strips_brackets(self):
        assert "white christmas" == normalize_title("White Christmas [Special Edition]")

    def test_strips_non_year_parens(self):
        assert "white christmas" == normalize_title("White Christmas (Widescreen)")

    def test_strips_trailing_year(self):
        assert "white christmas" == normalize_title("White Christmas (1954)")

    def test_html_entities(self):
        result = normalize_title("Nausicaa of the Valley of the Wind &amp; More")
        assert "and" in result
        assert "&amp;" not in result

    def test_strips_edition_keywords(self):
        assert "matrix" == normalize_title("The Matrix Remastered Limited Edition")

    def test_strips_disc_count(self):
        result = normalize_title("Lord of the Rings 2 Disc Set")
        assert "disc" not in result
        assert "lord" in result

    def test_normalizes_punctuation(self):
        result = normalize_title("Batman: The Dark Knight")
        assert ":" not in result

    def test_strips_leading_article_the(self):
        assert "matrix" == normalize_title("The Matrix")

    def test_strips_leading_article_a(self):
        assert "beautiful mind" == normalize_title("A Beautiful Mind")

    def test_strips_trailing_article(self):
        result = normalize_title("Good, The")
        assert "good" in result

    def test_region_free(self):
        assert "amelie" == normalize_title("Amelie Region Free")

    def test_ampersand_to_and(self):
        result = normalize_title("Tom & Jerry")
        assert "tom and jerry" == result

    def test_empty_string(self):
        assert "" == normalize_title("")

    def test_complex_real_world(self):
        title = "Alice in Wonderland Walt Disney Masterpiece Collection VHS"
        result = normalize_title(title)
        assert "alice in wonderland" in result
        assert "vhs" not in result


# ── clean_name ──

class TestCleanName:
    def test_decodes_html(self):
        result = clean_name("Gor&ocirc; Naya")
        assert "&ocirc;" not in result
        assert "Naya" in result

    def test_strips_whitespace(self):
        assert "Matrix" == clean_name("  Matrix  ")


# ── build_alias_map ──

class TestBuildAliasMap:
    def test_groups_duplicates(self):
        item_map = {
            "A1": "White Christmas VHS",
            "A2": "White Christmas DVD",
            "A3": "White Christmas",
        }
        alias_map, primary_names = build_alias_map(item_map)

        assert len(primary_names) == 1
        primary_asin = list(primary_names.keys())[0]
        assert primary_asin in ("A1", "A2", "A3")

        for asin in ("A1", "A2", "A3"):
            resolved = resolve(asin, alias_map)
            assert resolved == primary_asin

    def test_no_duplicates(self):
        item_map = {"A1": "The Matrix", "A2": "Inception"}
        alias_map, primary_names = build_alias_map(item_map)
        assert len(primary_names) == 2
        assert len(alias_map) == 0

    def test_picks_shortest_name_as_primary(self):
        item_map = {
            "A1": "White Christmas VHS",
            "A2": "White Christmas DVD",
            "A3": "White Christmas",
        }
        alias_map, primary_names = build_alias_map(item_map)
        assert len(primary_names) == 1
        primary_asin = list(primary_names.keys())[0]
        assert primary_names[primary_asin] == "White Christmas"


# ── resolve ──

class TestResolve:
    def test_resolves_alias(self):
        assert "A1" == resolve("A2", {"A2": "A1"})

    def test_returns_self_if_no_alias(self):
        assert "A1" == resolve("A1", {"A3": "A2"})

    def test_empty_alias_map(self):
        assert "A1" == resolve("A1", {})


# ── get_movie_name ──

class TestGetMovieName:
    def test_returns_name(self):
        item_map = {"A1": "The Matrix"}
        assert "The Matrix" == get_movie_name("A1", item_map, {})

    def test_resolves_alias_then_looks_up(self):
        item_map = {"A1": "The Matrix"}
        alias_map = {"A2": "A1"}
        assert "The Matrix" == get_movie_name("A2", item_map, alias_map)

    def test_returns_asin_if_not_found(self):
        assert "UNKNOWN" == get_movie_name("UNKNOWN", {}, {})

    def test_cleans_html_in_name(self):
        item_map = {"A1": "Gor&ocirc; Naya"}
        result = get_movie_name("A1", item_map, {})
        assert "&ocirc;" not in result


# ── get_user_history_names ──

class TestGetUserHistoryNames:
    def test_deduplicates(self):
        profile = {"history": ["A1", "A2", "A1"]}
        item_map = {"A1": "Matrix", "A2": "Inception"}
        names = get_user_history_names(profile, item_map, {})
        assert len(names) == 2
        assert names[0] == "Matrix"

    def test_resolves_aliases_before_dedup(self):
        profile = {"history": ["A1", "A2"]}
        item_map = {"A1": "Matrix"}
        alias_map = {"A2": "A1"}
        names = get_user_history_names(profile, item_map, alias_map)
        assert len(names) == 1

    def test_empty_history(self):
        profile = {"history": []}
        names = get_user_history_names(profile, {}, {})
        assert names == []


# ── search_movies ──

class TestSearchMovies:
    def test_finds_matching_movies(self):
        item_map = {"A1": "The Matrix", "A2": "Inception", "A3": "Matrix Reloaded"}
        results = search_movies("matrix", item_map, {}, limit=10)
        titles = [name for _, name in results]
        assert "The Matrix" in titles
        assert "Matrix Reloaded" in titles
        assert "Inception" not in titles

    def test_respects_limit(self):
        item_map = {f"A{i}": f"Star Wars {i}" for i in range(20)}
        results = search_movies("star wars", item_map, {}, limit=5)
        assert len(results) <= 5

    def test_deduplicates_aliases(self):
        item_map = {"A1": "Matrix", "A2": "Matrix VHS"}
        alias_map = {"A2": "A1"}
        results = search_movies("matrix", item_map, alias_map, limit=10)
        assert len(results) == 1

    def test_no_results(self):
        item_map = {"A1": "The Matrix"}
        results = search_movies("inception", item_map, {}, limit=10)
        assert results == []


# ── load_all (integration with real data) ──

class TestLoadAll:
    @pytest.fixture(scope="class")
    def data(self):
        return load_all()

    def test_item_map_loaded(self, data):
        assert len(data['item_map']) > 9000

    def test_profiles_loaded(self, data):
        assert len(data['profiles']) > 3000

    def test_conversations_loaded(self, data):
        assert len(data['conversations']) > 1000

    def test_alias_map_built(self, data):
        assert len(data['alias_map']) > 0

    def test_primary_names_built(self, data):
        assert len(data['primary_names']) > 0
        assert len(data['primary_names']) < len(data['item_map'])

    def test_user_index_correct(self, data):
        for uid, idx in data['user_index'].items():
            assert data['profiles'][idx]['user_id'] == uid

    def test_profile_has_required_keys(self, data):
        p = data['profiles'][0]
        assert 'user_id' in p
        assert 'history' in p
        assert 'might_like' in p
        assert 'conversations' in p

    def test_session_has_required_keys(self, data):
        p = data['profiles'][0]
        if p['conversations']:
            sess = p['conversations'][0]
            assert 'conversation_id' in sess
            assert 'user_likes' in sess
            assert 'user_dislikes' in sess
            assert 'rec_item' in sess
