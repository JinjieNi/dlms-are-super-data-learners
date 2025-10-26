#!/usr/bin/env python
# coding: utf-8

"""
This script scans a given directory for all files ending with "_document.bin"
and uses a custom IndexedDataset (that does not memory-map the .idx file) to
count the total number of tokens across all matched files.

By default, Megatron-LM memory-maps the .idx file via numpy.memmap, which can
fail on certain network-mounted systems. Here, we introduce a new class
IndexedDatasetCustom that loads .idx fully into normal memory (avoiding memmap).
We then use standard Python file I/O for both the .idx and .bin files.
This custom variant is used only here for counting and does not affect other files.
"""

import os
import argparse
import json
import sys

PROJECT_DIR = os.environ.get("PROJECT_DIR")
sys.path.append(PROJECT_DIR)
from megatron.core.datasets.indexed_dataset import IndexedDataset


def get_datadirs_and_weights(weighted_prefix: str):
    weighted_prefix_list = weighted_prefix.split(" ")
    weights = []
    datadirs = []
    for i, prefix in enumerate(weighted_prefix_list):
        if i % 2 == 0:
            # validate it's a float number
            if not prefix.replace('.', '').isdigit():
                raise ValueError(f"Weight {prefix} is not a valid float number")
            prefix = float(prefix)
            weights.append(prefix)
        else:
            data_dir = os.path.dirname(prefix)
            if not os.path.exists(data_dir):
                raise ValueError(f"Data directory {data_dir} does not exist")
            datadirs.append(prefix)
            
    assert len(weights) == len(datadirs)
    return zip(weights, datadirs)
        
def inspect_index(path_prefix: str):
    """
    Given a path prefix (without '.bin' or '.idx'), load the indexed dataset
    and print out some basic information such as total number of sequences
    and total number of tokens.

    Args:
        path_prefix (str): The file path prefix where <prefix>.bin and <prefix>.idx exist.
    """
    # Create the IndexedDataset from the path prefix.
    dataset = IndexedDataset(path_prefix)

    # The dataset size (number of sequences).
    total_sequences = len(dataset)
    

    # The total number of tokens across all sequences.
    total_tokens = dataset.sequence_lengths.sum()
    

    return total_sequences,total_tokens
    

def main():
    """
    Count tokens from all tokenized (_document.bin) files in a directory
    using IndexedDatasetCustom. This avoids memory-mapping the index file.
    """
    parser = argparse.ArgumentParser(
        description="Count tokens from all tokenized (_document.bin) files using a custom dataset."
    )
    parser.add_argument(
        "--weighted-prefix",
        type=str,
        required=True,
        help="A string as specified in the --data-path argument in megatron.training.arguments.py. Must be a list of weight prefix pairs e.g. weight1 prefix1 weight2 prefix2"
    )
    args = parser.parse_args()
    
    weights_and_prefixes = get_datadirs_and_weights(args.weighted_prefix)
    
    total_tokens_overall = []
    total_sequences_overall = []
    weights_overall = []
    
    for wp_pair in weights_and_prefixes:
        weight, prefix = wp_pair
        
        data_dir = os.path.dirname(prefix)
        statistics_file = os.path.join(data_dir, f".statistics_{prefix.replace('/', '_')}")
        
        if os.path.exists(statistics_file):
            with open(statistics_file, "r") as f:
                statistics = json.load(f)
                total_tokens = int(statistics["total_tokens"])
                total_sequences = int(statistics["total_sequences"])
        else:
            total_sequences, total_tokens = inspect_index(prefix)
            statistics = {
                "total_sequences": str(total_sequences),
                "total_tokens": str(total_tokens)
            }
            with open(statistics_file, "w") as f:
                json.dump(statistics, f)
        
        total_sequences_overall.append(total_sequences)
        total_tokens_overall.append(total_tokens)
        weights_overall.append(weight)
        
        print("--------------------------------")
        print(f"Inspecting index for {prefix}")
        print(f"Total number of sequences: {total_sequences}")
        print(f"Total number of tokens: {round(total_tokens / 1e9, 2)} Billion")
        print("--------------------------------")
    
    max_possible_tokens = min([token_num / weight for token_num, weight in zip(total_tokens_overall, weights_overall)])
    max_possible_sequences = min([seq_num / weight for seq_num, weight in zip(total_sequences_overall, weights_overall)])
    
    print("--------------------------------") 
    print(f"Maximum number of sequences sampled under this data mixture (estimated): {max_possible_sequences}")
    print(f"Maximum number of tokens sampled under this data mixture (estimated): {round(max_possible_tokens / 1e9, 2)} Billion")
    print("--------------------------------")
    
if __name__ == "__main__":
    main()