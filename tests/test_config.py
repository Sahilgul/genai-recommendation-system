from config import BASE_DIR, DATA_DIR, QWEN_MODEL, EMBEDDING_MODEL, EMBEDDING_BASE_URL


class TestConfig:
    def test_base_dir_is_submit_folder(self):
        assert BASE_DIR.name == "Submit_Folder"

    def test_data_dir_path(self):
        assert DATA_DIR == BASE_DIR / "LLM_Redial" / "Movie"

    def test_qwen_model_set(self):
        assert QWEN_MODEL and isinstance(QWEN_MODEL, str)

    def test_embedding_model_set(self):
        assert EMBEDDING_MODEL and isinstance(EMBEDDING_MODEL, str)

    def test_embedding_base_url_set(self):
        assert EMBEDDING_BASE_URL and EMBEDDING_BASE_URL.startswith("http")
