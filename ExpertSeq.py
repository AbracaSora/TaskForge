import os
import json


DATA_DIR = "./action_list_single"

files = os.listdir(DATA_DIR)


def get_expert_seqs():
    expert_seqs = []
    for file in files:
        with open(os.path.join(DATA_DIR, file), "r") as f:
            expert_seq = json.load(f)
            expert_seqs.append(expert_seq["actions"])
    return expert_seqs
