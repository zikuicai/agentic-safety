import numpy as np
# import json
import re
import os


# def clean_json(json_str: str) -> str:
#     Remove the trailing comma before the closing brace
    # cleaned_json_str = re.sub(r',\s*}', '\n}', json_str)
    # cleaned_json_str = re.sub(r'#.*', '', cleaned_json_str)
    # return cleaned_json_str


if __name__ == '__main__':
    assert os.path.exists('mappings.npy')

    # d = list of strings in the format f'translations = {json_str}'
    d = np.load('mappings.npy')

    all_mappings_str = '\n'.join(d)

    all_mappings = re.findall(r'"(.*?)"\s*:\s*"(.*?)"', all_mappings_str)
    # print(all_mappings, len(all_mappings))

    all_unlearned_names = list(set([x[0] for x in all_mappings]))
    all_unlearned_names.sort()
    print(all_unlearned_names, len(all_unlearned_names))

    with open('../unlearning_harry_potter.txt', 'w') as f:
        for name in all_unlearned_names:
            f.write(f'{name}\n')


