import random

from data_loader import resolve
from few_shot.prompts import build_prompt
from llm import stream_chat


def find_few_shot_examples(profile, data, n=2):
    user_history = set(resolve(a, data['alias_map']) for a in profile['history'])
    scored = []

    for other in data['profiles']:
        if other['user_id'] == profile['user_id']:
            continue
        other_history = set(resolve(a, data['alias_map']) for a in other['history'])
        overlap = len(user_history & other_history)
        if overlap > 0 and other['conversations']:
            scored.append((overlap, other))

    scored.sort(key=lambda x: x[0], reverse=True)

    examples = []
    for _score, other in scored[:n * 2]:
        for sess in other['conversations']:
            cid = sess['conversation_id']
            if cid in data['conversations']:
                text = data['conversations'][cid]
                if text and len(text) > 50:
                    examples.append(text.strip())
                    if len(examples) >= n:
                        break
        if len(examples) >= n:
            break

    if len(examples) < n:
        all_ids = list(data['conversations'].keys())
        random.shuffle(all_ids)
        for cid in all_ids:
            text = data['conversations'][cid]
            if text and len(text) > 50 and text.strip() not in examples:
                examples.append(text.strip())
                if len(examples) >= n:
                    break

    return examples[:n]


async def stream_recommendation(profile, data, chat_history, user_message):
    few_shots = find_few_shot_examples(profile, data)
    system = build_prompt(profile, data, few_shots)

    messages = [{"role": "system", "content": system}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_message})

    async for token in stream_chat(messages):
        yield token
