from evaluate import extract_movie_titles


def make_item_map():
    return {
        "A1": "The Matrix",
        "A2": "Inception",
        "A3": "Interstellar",
        "A4": "The Dark Knight",
        "A5": "Pulp Fiction",
        "A6": "The Matrix VHS",
    }


def make_alias_map():
    return {"A6": "A1"}


class TestExtractMovieTitles:
    def test_finds_quoted_movie(self):
        response = 'I recommend "The Matrix" for you!'
        titles = extract_movie_titles(response, make_item_map(), make_alias_map())
        assert "A1" in titles

    def test_finds_unquoted_movie(self):
        response = "You should watch Inception, it is great."
        titles = extract_movie_titles(response, make_item_map(), make_alias_map())
        assert "A2" in titles

    def test_finds_multiple_movies(self):
        response = 'Try "The Matrix" and "Inception" and maybe Interstellar.'
        titles = extract_movie_titles(response, make_item_map(), make_alias_map())
        assert len(titles) >= 3

    def test_deduplicates_results(self):
        response = '"The Matrix" is great. The Matrix is a classic.'
        titles = extract_movie_titles(response, make_item_map(), make_alias_map())
        assert titles.count("A1") == 1

    def test_resolves_aliases(self):
        response = 'Watch "The Matrix VHS" edition.'
        titles = extract_movie_titles(response, make_item_map(), make_alias_map())
        assert "A1" in titles
        assert "A6" not in titles

    def test_no_matches(self):
        response = "I have no recommendations today."
        titles = extract_movie_titles(response, make_item_map(), make_alias_map())
        assert titles == []

    def test_skips_short_names(self):
        item_map = {"X1": "IT", "X2": "Inception"}
        response = "It is a great movie. I recommend Inception."
        titles = extract_movie_titles(response, item_map, {})
        assert "X2" in titles
