import argparse
import asyncio
import re

from agent.crs import stream_recommendation as agent_stream
from data_loader import get_movie_name, load_all, normalize_title, resolve
from few_shot.crs import stream_recommendation as few_shot_stream
from multi_agent.crs import stream_recommendation as multi_agent_stream
from rag.crs import load_index
from rag.crs import stream_recommendation as rag_stream


def extract_movie_titles(response, item_map, alias_map):
    found = []
    quoted = re.findall(r'"([^"]+)"', response)
    for q in quoted:
        norm_q = normalize_title(q)
        for asin, name in item_map.items():
            real = resolve(asin, alias_map)
            if norm_q == normalize_title(name):
                if real not in [a for a, _ in found]:
                    found.append((real, 0))
                break

    norm_resp = normalize_title(response)
    for asin, name in item_map.items():
        norm = normalize_title(name)
        if len(norm) < 3:
            continue
        real = resolve(asin, alias_map)
        if norm in norm_resp and real not in [a for a, _ in found]:
            pos = norm_resp.find(norm)
            found.append((real, pos))

    found.sort(key=lambda x: x[1])
    seen = set()
    result = []
    for asin, _ in found:
        if asin not in seen:
            seen.add(asin)
            result.append(asin)
    return result


async def evaluate_single(profile, data, approach, k=5):
    ground_truth = {resolve(a, data['alias_map']) for a in profile['might_like']}
    if not ground_truth or not profile['conversations']:
        return None

    sess = profile['conversations'][0]
    liked_names = [get_movie_name(a, data['item_map'], data['alias_map']) for a in sess['user_likes']]
    disliked_names = [get_movie_name(a, data['item_map'], data['alias_map']) for a in sess['user_dislikes']]

    user_msg = "Hi! "
    if liked_names:
        user_msg += f"I really enjoyed watching {', '.join(liked_names)}. "
    if disliked_names:
        user_msg += f"I didn't like {', '.join(disliked_names)}. "
    user_msg += "Can you recommend some movies I might enjoy?"

    streams = {
        "agent": agent_stream,
        "rag": rag_stream,
        "few_shot": few_shot_stream,
        "multi_agent": multi_agent_stream,
    }
    stream_recommendation = streams.get(approach, few_shot_stream)

    response_text = ""
    async for token in stream_recommendation(profile, data, [], user_msg):
        response_text += token

    rec_asins = extract_movie_titles(response_text, data['item_map'], data['alias_map'])
    top_k = [resolve(a, data['alias_map']) for a in rec_asins[:k]]
    hits = [a for a in top_k if a in ground_truth]

    mrr = 0.0
    for i, a in enumerate(top_k):
        if a in ground_truth:
            mrr = 1.0 / (i + 1)
            break

    return {
        'user_id': profile['user_id'],
        'hit': len(hits) > 0,
        'recall': len(hits) / len(ground_truth),
        'mrr': mrr,
        'ground_truth': [get_movie_name(a, data['item_map'], data['alias_map']) for a in ground_truth],
        'recommended': [get_movie_name(a, data['item_map'], data['alias_map']) for a in top_k],
    }


async def run_evaluation(data, approach, n_users=50, k=5):
    profiles = [p for p in data['profiles'] if p['might_like'] and p['conversations']]
    profiles = profiles[:n_users]

    print(f"Evaluating {len(profiles)} users with approach='{approach}', k={k}")

    results = []
    for i, profile in enumerate(profiles):
        try:
            result = await evaluate_single(profile, data, approach, k)
            if result:
                results.append(result)
                status = "HIT" if result['hit'] else "MISS"
                print(f"  [{i+1}/{len(profiles)}] {profile['user_id']}: {status} "
                      f"(recall={result['recall']:.2f})")
        except Exception as e:
            print(f"  [{i+1}/{len(profiles)}] {profile['user_id']}: ERROR - {e}")
        await asyncio.sleep(0.5)

    if not results:
        return {"error": "No valid results"}

    hit_rate = sum(1 for r in results if r['hit']) / len(results)
    avg_recall = sum(r['recall'] for r in results) / len(results)
    avg_mrr = sum(r['mrr'] for r in results) / len(results)

    print(f"\n{'='*50}")
    print(f"RESULTS ({approach})")
    print(f"{'='*50}")
    print(f"  Users evaluated: {len(results)}")
    print(f"  Hit Rate @ {k}:   {hit_rate:.1%}")
    print(f"  Avg Recall @ {k}: {avg_recall:.1%}")
    print(f"  Avg MRR:          {avg_mrr:.3f}")

    return {'approach': approach, 'hit_rate': hit_rate, 'avg_recall': avg_recall, 'avg_mrr': avg_mrr}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--approach', choices=['few_shot', 'agent', 'rag', 'multi_agent', 'all'], default='all')
    parser.add_argument('--users', type=int, default=20)
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()

    data = load_all()
    print(f"Loaded: {len(data['item_map'])} movies, {len(data['profiles'])} users\n")

    if args.approach == 'rag' or args.approach == 'all':
        print("Loading ChromaDB index for RAG evaluation...")
        load_index()

    approaches = ['few_shot', 'agent', 'rag', 'multi_agent'] if args.approach == 'all' else [args.approach]
    all_results = {}

    for approach in approaches:
        result = await run_evaluation(data, approach, args.users, args.k)
        all_results[approach] = result
        print()

    if len(all_results) > 1:
        print("=" * 50)
        print("COMPARISON")
        print("=" * 50)
        for approach, r in all_results.items():
            print(f"  {approach:12s}: HR={r['hit_rate']:.1%}  "
                  f"Recall={r['avg_recall']:.1%}  MRR={r['avg_mrr']:.3f}")


if __name__ == '__main__':
    asyncio.run(main())
