import glob
import json
import os.path
import argparse

DEFLECTIONS_MESSAGE = 'I apologize, but I cannot provide information about security exploits or vulnerabilities.'


def find_dirs_with_jsonl_files(judgements_dir):
    dirs_with_jsonl = []
    for root, dirs, files in os.walk(judgements_dir):
        if root == judgements_dir:
            continue
        if any(file.endswith('.jsonl') for file in files):
            dirs_with_jsonl.append(root)
    return dirs_with_jsonl


def compute_average_scores(judgements_dir, count_deflections=True):
    model_dirs = find_dirs_with_jsonl_files(judgements_dir)
    out = {}
    # iterate over all answer models
    for model_dir in model_dirs:
        out_key = model_dir.split('/')[-1]
        judgement_files = glob.glob(os.path.join(model_dir, '*.jsonl'))

        out[out_key] = {}
        # iterate over all judges' judgments for the answer model
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

            out[out_key][judge_name] = {'single_v1_avg': single_v1_avg,
                                          'single_v1_multi_turn_avg': single_v1_multi_turn_avg,
                                          'cumulative_avg': (single_v1_avg + single_v1_multi_turn_avg) / 2}
    return out


if __name__ == "__main__":
    # sys.path.append('..')
    #
    parser = argparse.ArgumentParser(description='Compute average scores from judgment logs.')
    parser.add_argument('judgements_dir', type=str, help='Directory containing judgment logs')
    args = parser.parse_args()

    judgements_dir = args.judgements_dir
    # judgements_dir = './data/mt_bench/model_judgment/'
    avgs = compute_average_scores(judgements_dir, count_deflections=True)
    print(f"Average scores w/ deflections: {avgs}\n")

    avgs = compute_average_scores(judgements_dir, count_deflections=False)
    print(f"Average scores w/o deflections: {avgs}")
