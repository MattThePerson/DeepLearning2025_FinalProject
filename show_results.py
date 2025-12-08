import argparse
from pathlib import Path
import os
from datetime import datetime

import pandas as pd
import numpy as np


def main(args: argparse.Namespace):
    
    os.chdir(args.results_dir)
    
    # read files from results_dir
    csv_files = Path('.').rglob("*.csv")
    dataframes = {
        str(fn).replace('.csv', '.pt'): pd.read_csv(str(fn)).set_index("Category")
        for fn in csv_files
    }
    
    # 
    print('viewing category:', args.category)
    table = pd.DataFrame()
    for fn, df in dataframes.items():
        series = df.T[args.category]
        pt_file = f"../{args.checkpoints_dir}/{fn}"
        ctime = os.path.getctime(pt_file)
        series["date_created"] = datetime.fromtimestamp(ctime).strftime("%Y-%m-%d %H:%M:%S")
        table[fn] = series
    
    table = table.T
    print('sorting checkpoints by:', args.sort)
    table = table.sort_values(args.sort, ascending=args.reverse) # type: ignore
    table = table.reset_index().rename(columns={"index": "checkpoint"})
    table.index = table.index + 1
    
    print("\n=========================")
    print("CHECKPOINT METRICS")
    print(table)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--category", default="Average", choices=["DR", "Glaucoma", "AMD", "Average"])
    parser.add_argument("--sort", default="F1-score")
    parser.add_argument("--reverse", action='store_true')
    parser.add_argument("--results_dir", default="test_results")
    parser.add_argument("--checkpoints_dir", default="checkpoints") # to get date created for checkpoint
    
    args = parser.parse_args()
    
    print()
    main(args)
    print()