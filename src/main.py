import argparse
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_scipy_sparse_matrix
import torch
import numpy as np
import sys
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import csv
import time
from datetime import datetime
from utils import load_dataset, shuffle_data
from model import Congrat, train, test

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7, help='Random seed.')
    parser.add_argument('--dataset', type=str, nargs='+', default=['COVID19'], help="One or more of ['COVID19', 'FakeNewsNet', 'Liar', 'PAN2020'], run sequentially in one invocation")
    # new add
    parser.add_argument("--batch_size", type=int, default=200, help='set the batch size of the training data into our models')
    # parser.add_argument("--kg_sel", type=int, defalut=1, help="Using this parameter can choose different KG.")
    parser.add_argument("--alpha", type=float, default=1, help="The alpha hyperparameter to be adjusted for training loss. [0.1-2.0]")

    # GNN related parameters
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.default=200,[50, 100, 150, 200, 300]')
    parser.add_argument('--hidden_channels', type=int, default=256, help='Dim of 1st layer GNN. 32,64,128,256, default=256')
    parser.add_argument('--gnn_layers', type=int, default=2, help='Number of GNN layers. 2,3, default=2')
    parser.add_argument('--learning_rate', default=0.0005, help='Learning rate of the optimiser. 0.0001, 0.001, default=0.0005')
    parser.add_argument('--weight_decay', default=5e-4, help='Weight decay of the optimiser. default=5e-4')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # empty cuda allocation
    # torch.cuda.empty_cache()
    
    args = arg_parser()
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset_list = args.dataset
    csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results.csv")
    per_dataset_time = {}

    for dataset_name in dataset_list:
        args.dataset = dataset_name

        print(f"\n{'#'*55}\n# Dataset: {dataset_name}\n{'#'*55}")
        print("loading data")
        start_total_time = time.time()
        hgraph = load_dataset(args.dataset)
        args.device = device
        random.seed(args.seed)
        hgraph = shuffle_data(hgraph, args)

        acc_list, prec_list, rec_list, f1_list = [], [], [], []
        for i in range(10):
            print(f"\n--- Run {i+1}/10 ---")

            # KHÓA HẠT GIỐNG NGẪU NHIÊN (Khử nhiễu đồ thị Liar)
            seed = args.seed + i
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            model = Congrat(hidden_channels=args.hidden_channels, out_channels=2, num_layers=args.gnn_layers, dropout_rate=args.dropout)
            model.to(device)
            hgraph.to(device)

            # Initialize parameters via lazy initialization
            with torch.no_grad():  # Initialize lazy modules.
                _, _, _, out = model(hgraph.x_dict, hgraph.edge_index_dict)

            train(model, hgraph, args)

            model.eval()
            with torch.no_grad():
                acc, prec, rec, f1 = test(model, hgraph, args)
                acc_list.append(acc)
                prec_list.append(prec)
                rec_list.append(rec)
                f1_list.append(f1)

        # IN KẾT QUẢ TỔNG HỢP (FINAL RESULTS)
        print("\n" + "="*55)
        print(f"FINAL RESULTS AFTER 10 RUNS - {dataset_name} (BẢNG KẾT QUẢ TỔNG HỢP)")
        print("="*55)
        print(f"{'Metric':<12} | {'Mean':<8} | {'Std':<8} | {'Min':<8} | {'Max':<8}")
        print("-" * 55)

        def print_stat(name, arr):
            mean, std, mn, mx = np.mean(arr), np.std(arr), np.min(arr), np.max(arr)
            print(f"{name:<12} | {mean:.4f}   | {std:.4f}   | {mn:.4f}   | {mx:.4f}")
            return [name, f"{mean:.4f}", f"{std:.4f}", f"{mn:.4f}", f"{mx:.4f}"]

        results_data = []
        results_data.append(print_stat("Accuracy", acc_list))
        results_data.append(print_stat("Precision", prec_list))
        results_data.append(print_stat("Recall", rec_list))
        results_data.append(print_stat("F1-Score", f1_list))

        end_total_time = time.time()
        total_time_seconds = end_total_time - start_total_time
        total_time_minutes = total_time_seconds / 60
        per_dataset_time[dataset_name] = total_time_seconds
        print("-" * 55)
        print(f"Total Execution Time ({dataset_name}): {total_time_seconds:.2f} seconds ({total_time_minutes:.2f} minutes)")
        print("="*55 + "\n")

        # Lưu kết quả ra file CSV
        file_exists = os.path.exists(csv_file)
        with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Timestamp', 'Dataset', 'Epochs', 'Learning Rate', 'Time (s)', 'Metric', 'Mean', 'Std', 'Min', 'Max'])

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for row in results_data:
                writer.writerow([timestamp, dataset_name, args.epochs, args.learning_rate, f"{total_time_seconds:.2f}"] + row)

        print(f"Kết quả đã được lưu vào file: {os.path.abspath(csv_file)}\n")

    if len(dataset_list) > 1:
        print("\n" + "="*55)
        print("TOTAL RUN TIME PER DATASET")
        print("="*55)
        for dataset_name, secs in per_dataset_time.items():
            print(f"{dataset_name:<12} | {secs:.2f}s ({secs/60:.2f} min)")
        print(f"{'TOTAL':<12} | {sum(per_dataset_time.values()):.2f}s ({sum(per_dataset_time.values())/60:.2f} min)")
        print("="*55 + "\n")
