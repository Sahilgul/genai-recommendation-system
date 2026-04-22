import html
import json
import re
from collections import defaultdict
from config import ITEM_MAP_PATH, USER_IDS_PATH, FINAL_DATA_PATH, CONVERSATION_PATH


def normalize_title(name):
    n = html.unescape(name).strip()
    for tag in ('VHS', 'DVD', 'Blu-ray', 'Blu-Ray', 'NTSC', 'PAL'):
        n = re.sub(rf'\b{tag}\b', '', n, flags=re.IGNORECASE)
    n = re.sub(r'\[.*?\]', '', n)
    n = re.sub(r'\((?!\d{4}\))[^)]*\)', '', n)
    n = re.sub(r'\(\d{4}\)\s*$', '', n)
    n = re.sub(r'region\s*(free|\d)', '', n, flags=re.IGNORECASE)
    n = re.sub(
        r'\b(edition|steelbook|collector|limited|import|widescreen|fullscreen'
        r'|letterbox|unrated|uncut|remastered|digitally|restored|THX)\b',
        '', n, flags=re.IGNORECASE,
    )
    n = re.sub(r'\d+\s*disc\w*', '', n, flags=re.IGNORECASE)
    n = re.sub(r'[:\-,/]', ' ', n)
    n = n.replace('&', 'and')
    n = re.sub(r'\s+', ' ', n).strip().strip('- /:,.')
    n = n.lower()
    n = re.sub(r'^(the|a|an)\s+', '', n)
    n = re.sub(r',\s*(the|a|an)\s*$', '', n)
    return n.strip()


def clean_name(name):
    return html.unescape(name).strip()


def build_alias_map(item_map):
    groups = defaultdict(list)
    for asin, name in item_map.items():
        groups[normalize_title(name)].append((asin, name))

    alias_map = {}
    primary_names = {}

    for core, entries in groups.items():
        best = min(entries, key=lambda x: len(clean_name(x[1])))
        primary = best[0]
        primary_names[primary] = clean_name(best[1])
        for asin, _ in entries:
            if asin != primary:
                alias_map[asin] = primary

    return alias_map, primary_names


def resolve(asin, alias_map):
    return alias_map.get(asin, asin)


def load_item_map():
    with open(ITEM_MAP_PATH, encoding='utf-8') as f:
        return json.load(f)


def load_user_ids():
    with open(USER_IDS_PATH, encoding='utf-8') as f:
        return json.load(f)


def load_profiles():
    profiles = []
    with open(FINAL_DATA_PATH, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for user_id, data in record.items():
                sessions = []
                for conv_entry in data.get('Conversation', []):
                    for _key, sess in conv_entry.items():
                        sessions.append({
                            'conversation_id': sess['conversation_id'],
                            'user_likes': sess.get('user_likes', []),
                            'user_dislikes': sess.get('user_dislikes', []),
                            'rec_item': sess.get('rec_item', []),
                        })
                profiles.append({
                    'user_id': user_id,
                    'history': data['history_interaction'],
                    'might_like': data['user_might_like'],
                    'conversations': sessions,
                })
    return profiles


def load_conversations():
    convos = {}
    with open(CONVERSATION_PATH, encoding='utf-8') as f:
        content = f.read()
    chunks = re.split(r'\n(?=\d+\n)', content)
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        lines = chunk.split('\n', 1)
        if lines[0].strip().isdigit():
            cid = int(lines[0].strip())
            text = lines[1].strip() if len(lines) > 1 else ''
            convos[cid] = text
    return convos


def get_movie_name(asin, item_map, alias_map):
    real = resolve(asin, alias_map)
    name = item_map.get(real, item_map.get(asin, asin))
    return clean_name(name) if name != asin else asin


def get_user_history_names(profile, item_map, alias_map):
    seen = set()
    names = []
    for asin in profile['history']:
        real = resolve(asin, alias_map)
        if real not in seen:
            seen.add(real)
            names.append(get_movie_name(real, item_map, alias_map))
    return names


def search_movies(query, item_map, alias_map, limit=10):
    q = normalize_title(query)
    results = []
    seen = set()
    for asin, name in item_map.items():
        real = resolve(asin, alias_map)
        if real in seen:
            continue
        if q in normalize_title(name):
            seen.add(real)
            results.append((real, clean_name(name)))
    return results[:limit]


def load_all():
    item_map = load_item_map()
    load_user_ids()
    alias_map, primary_names = build_alias_map(item_map)
    profiles = load_profiles()
    conversations = load_conversations()

    user_index = {p['user_id']: i for i, p in enumerate(profiles)}

    return {
        'item_map': item_map,
        'alias_map': alias_map,
        'primary_names': primary_names,
        'profiles': profiles,
        'conversations': conversations,
        'user_index': user_index,
    }


if __name__ == '__main__':
    data = load_all()
    print(f"Movies: {len(data['item_map'])}")
    print(f"Unique: {len(data['primary_names'])}")
    print(f"Aliases: {len(data['alias_map'])}")
    print(f"Users: {len(data['profiles'])}")
    print(f"Conversations: {len(data['conversations'])}")

    p = data['profiles'][0]
    history = get_user_history_names(p, data['item_map'], data['alias_map'])
    print(f"\nUser {p['user_id']} history ({len(p['history'])} raw → {len(history)} deduped):")
    for name in history[:5]:
        print(f"  - {name}")
