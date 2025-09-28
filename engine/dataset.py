import json
from typing import List
from termcolor import colored


def load_sharegpt_prompts(path: str, max_lines: int = 10000, min_len: int = 10, max_len: int = 500) -> List[str]:
    """Load human prompts from a ShareGPT jsonl file.

    - path: jsonl path
    - max_lines: read at most this many lines to limit IO
    - min_len/max_len: filter prompt lengths
    """
    prompts: List[str] = []
    try:
        print(colored("Loading ShareGPT data...", "light_grey"))
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                conv = data.get("conversations")
                if not conv:
                    continue
                for turn in conv:
                    if turn.get("from") == "human" and turn.get("value"):
                        text = turn["value"].strip()
                        if min_len <= len(text) <= max_len:
                            prompts.append(text)
        print(colored(f"Loaded {len(prompts)} prompts from ShareGPT", "green"))
    except FileNotFoundError:
        print(colored(f"ShareGPT file not found: {path}", "red"))
    return prompts

def load_sharegpt_multi(paths: List[str], max_lines: int = 10000, min_len: int = 10, max_len: int = 500) -> List[List[str]]:
    """Load prompts from multiple ShareGPT jsonl files.

    Returns a list with one list of prompts per input path. Missing files yield empty lists.
    """
    out: List[List[str]] = []
    for p in paths:
        out.append(load_sharegpt_prompts(p, max_lines=max_lines, min_len=min_len, max_len=max_len))
    return out


