import glob
import json
import os.path
import sys

DEFLECTIONS_MESSAGE = 'I apologize, but I cannot provide information about security exploits or vulnerabilities.'


def compute_average_scores(judgements_dir, count_deflections=True):
    judgement_files = glob.glob(os.path.join(judgements_dir, '*.jsonl'))

    out = {}
    for file_path in judgement_files:
        single_v1_scores = []
        single_v1_multi_turn_scores = []
        judge_name = ''.join(os.path.basename(file_path).split('.')[:-1])

        with open(file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                if not count_deflections and DEFLECTIONS_MESSAGE in data['user_prompt']:
                    continue
                if 'single-v1' in data['judge'] or 'single-math-v1' in data['judge']:
                    single_v1_scores.append(data['score'])
                if 'single-v1-multi-turn' in data['judge'] or 'single-math-v1-multi-turn' in data['judge']:
                    single_v1_multi_turn_scores.append(data['score'])

        single_v1_avg = sum(single_v1_scores) / len(single_v1_scores) if single_v1_scores else -1
        single_v1_multi_turn_avg = sum(single_v1_multi_turn_scores) / len(
            single_v1_multi_turn_scores) if single_v1_multi_turn_scores else -1

        out[judge_name] = {'single_v1_avg': single_v1_avg, 'single_v1_multi_turn_avg': single_v1_multi_turn_avg, 'cumulative_avg': (single_v1_avg + single_v1_multi_turn_avg) / 2}
    return out


if __name__ == "__main__":
    sys.path.append('..')

    judgements_dir = './data/mt_bench/model_judgment/Qwen'
    avgs = compute_average_scores(judgements_dir, count_deflections=True)
    print(f"Average scores w/ deflections: {avgs}\n")

    avgs = compute_average_scores(judgements_dir, count_deflections=False)
    print(f"Average scores w/o deflections: {avgs}")
