"""
llm-jp/magpie-sft-v1.0: native Japanese instruction/conversation dataset
generated via the Magpie method (cyberagent/calm3-22b-chat generates the
user turns, Qwen2.5-32B-Instruct generates the assistant turns). Apache-2.0.
https://huggingface.co/datasets/llm-jp/magpie-sft-v1.0
"""

from tasks.common import Task, load_hub_dataset

class MagpieJa(Task):
    """ ~132K rows, single train split (no official test split). """

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "MagpieJa split must be train|test"
        ds = load_hub_dataset("llm-jp/magpie-sft-v1.0", split="train").shuffle(seed=42)
        # no official train/test split upstream: carve off the last 5% as a held-out test set
        n = len(ds)
        n_test = max(1, n // 20)
        if split == "train":
            self.ds = ds
            self.offset = 0
            self.length = n - n_test
        else:
            self.ds = ds
            self.offset = n - n_test
            self.length = n_test

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[self.offset + index]
        conversation = {"messages": row["conversations"]}
        return conversation
