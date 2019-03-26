import os
import json
import torch
import shlex
import argparse
import subprocess
import pandas as pd
from pathlib import Path
from itertools import product
from supernnova.utils import logging_utils as lu

LIST_SEED = [0]#, 100, 1000, 55, 30496]

def run_cmd(cmd, debug, seed):
    """Run command
    Using cuda if available
    """

    cmd += f" --seed {seed} "

    if torch.cuda.is_available():
        cmd += " --use_cuda "

    if debug is True:
        # Run for 1 epoch only
        # cmd += "--cyclic_phases 1 1 1 "
        cmd += "--nb_epoch 1 "


    subprocess.check_call(shlex.split(cmd))


def run_CNN(dump_dir, debug, seed):
    """Baseline/Random Forest Accuracy vs. number of supernovae
        Default configurations used when not specified
        e.g. source_data(saltfit),modelrnn(vanilla),norm(global)
    """

    lu.print_green(f"SEED {seed}: TRAINING")

    #################################
    # Train baseline models on SALT #
    #################################
    list_redshift = [None, "zpho", "zspe"]
    list_source_data = ["saltfit"]#,"photometry"]

    # Train RNN models
    for (source_data,redshift) in product(list_source_data,list_redshift):
        cmd = (
            f"python -W ignore run.py --train_rnn "
            f"--source_data {source_data} "
            f"--dump_dir {dump_dir} "
            f"--model CNN "
        )
        if redshift is not None:
            cmd += f" --redshift {redshift} "
        run_cmd(cmd, debug, seed)


    # #######################################
    # # Train baseline models on COMPLETE   #
    # # goal: multiclass                    #
    # #######################################

    # list_nb_classes = [2, 7]
    # for (redshift, nb_classes) in product(list_redshift, list_nb_classes):
    #     cmd = (
    #         f"python -W ignore run.py --train_rnn "
    #         f"--nb_classes {nb_classes} "
    #         f"--dump_dir {dump_dir} "
    #         f"--source_data photometry "
    #         f"--model CNN "
    #     )
    #     if redshift is not None:
    #         cmd += f" --redshift {redshift} "
    #     run_cmd(cmd, debug, seed)


def run_baseline_hp(dump_dir, debug, seed):

    lu.print_green(f"SEED {seed}: BASELINE HP")

    if seed != LIST_SEED[0]:
        return

    list_batch_size = [64, 128, 512]
    list_bidirectional = [True, False]
    list_random_length = [True, False]
    list_hidden_dim = [16, 32, 64]
    list_learning_rate = [1e-2,1e-3,1e-4]

    if debug is True:
        list_batch_size = list_batch_size[:1]
        list_hidden_dim = list_hidden_dim[:1]

    for (
        learning_rate,
        batch_size,
        bidirectional,
        random_length,
        hidden_dim,
    ) in product(
        list_learning_rate,
        list_batch_size,
        list_bidirectional,
        list_random_length,
        list_hidden_dim,
    ):
        cmd = (
            f"python -W ignore run.py --train_rnn "
            f"--model CNN "
            f"--dump_dir {dump_dir} "
            f"--cyclic "
            f"--data_fraction 0.2 "
            f"--batch_size {batch_size} "
            f"--bidirectional {bidirectional} "
            f"--random_length {random_length} "
            f"--hidden_dim {hidden_dim} "
            f"--learning_rate {learning_rate} "
        )
        run_cmd(cmd, debug, seed)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SNIa classification")

    dir_path = os.path.dirname(os.path.realpath(__file__))
    default_dump_dir = Path(dir_path).parent / "snndump"
    parser.add_argument(
        "--dump_dir",
        type=str,
        default=default_dump_dir,
        help="Default path where data and models are dumped",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Switch to debug mode: will run dummy experiments to quickly check the whole pipeline",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=LIST_SEED,
        nargs="+",
        choices=LIST_SEED,
        help="Seed with which to run the experiments",
    )
    args = parser.parse_args()

    list_seeds = args.seeds[:2] if args.debug else args.seeds

    for seed in list_seeds:
        # run_CNN(args.dump_dir, args.debug, seed)
        run_baseline_hp(args.dump_dir, args.debug, seed)