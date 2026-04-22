from few_shot.crs import find_few_shot_examples


def make_data():
    profiles = [
        {
            'user_id': 'U1',
            'history': ['A1', 'A2', 'A3'],
            'might_like': [],
            'conversations': [
                {'conversation_id': 1, 'user_likes': [], 'user_dislikes': [], 'rec_item': []},
            ],
        },
        {
            'user_id': 'U2',
            'history': ['A1', 'A2'],
            'might_like': [],
            'conversations': [
                {'conversation_id': 2, 'user_likes': [], 'user_dislikes': [], 'rec_item': []},
            ],
        },
        {
            'user_id': 'U3',
            'history': ['A4', 'A5'],
            'might_like': [],
            'conversations': [
                {'conversation_id': 3, 'user_likes': [], 'user_dislikes': [], 'rec_item': []},
            ],
        },
    ]
    conversations = {
        1: "User: I love action movies! Assistant: Great, try Die Hard.",
        2: "User: What about comedies? Assistant: Watch The Hangover. It is very funny and entertaining, a classic.",
        3: "User: Any horror? Assistant: Try The Shining by Kubrick. It is classic horror, genuinely scary and well made.",
    }
    return {
        'item_map': {'A1': 'M1', 'A2': 'M2', 'A3': 'M3', 'A4': 'M4', 'A5': 'M5'},
        'alias_map': {},
        'primary_names': {},
        'profiles': profiles,
        'conversations': conversations,
        'user_index': {'U1': 0, 'U2': 1, 'U3': 2},
    }


class TestFindFewShotExamples:
    def test_returns_requested_count(self):
        data = make_data()
        profile = data['profiles'][0]
        examples = find_few_shot_examples(profile, data, n=2)
        assert len(examples) == 2

    def test_prefers_overlapping_users(self):
        data = make_data()
        profile = data['profiles'][0]
        examples = find_few_shot_examples(profile, data, n=1)
        assert len(examples) == 1
        assert "What about comedies" in examples[0]

    def test_excludes_own_profile(self):
        data = make_data()
        profile = data['profiles'][0]
        examples = find_few_shot_examples(profile, data, n=1)
        assert len(examples) == 1
        assert examples[0] != data['conversations'][1].strip() or len(data['profiles']) == 1

    def test_falls_back_to_random_if_not_enough(self):
        data = make_data()
        data['profiles'] = [data['profiles'][0]]
        profile = data['profiles'][0]
        examples = find_few_shot_examples(profile, data, n=2)
        assert len(examples) == 2

    def test_skips_short_conversations(self):
        data = make_data()
        data['conversations'][2] = "short"
        profile = data['profiles'][0]
        examples = find_few_shot_examples(profile, data, n=1)
        for ex in examples:
            assert len(ex) > 50
