import os
import pandas as pd
from pathlib import Path
import supernnova.conf as conf
from . import superNNova_plots as sp
from . import superNNova_metrics as sm
from ..utils import logging_utils as lu
from ..visualization import early_prediction

"""
Obtaining metrics and plots for SuperNNova paper

Selection of models is hard coded

Code is far from optimized
"""

"""
Best performing algorithms in SuperNNova
"""
Base = (
    "CNN_CLF_2_R_zpho_photometry_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean_0.001"
)
list_models = [Base]
list_models_rnn = [Base]

# useful formats
Base_salt = Base.replace("photometry", "saltfit")


def SuperNNova_stats_and_plots(settings):
    """ Reproduce stats and plots used for SuperNNova paper.
    BEWARE: Selection is hardcoded

    Args:
        settings (ExperimentSettings): custom class to hold hyperparameters
    """

    # Load summary statistics
    df_stats = pd.read_csv(Path(settings.stats_dir) / "summary_stats.csv")

    # Create latex tables
    # sm.create_accuracy_latex_tables(df_stats, settings)

    # Rest of stats and plots in paper
    # can be ran in debug mode: only printing model names
    # or in no plot mode: only printing stats
    SuperNNova_stats_and_plots_thread(df_stats, settings, plots=True, debug=False)


def SuperNNova_stats_and_plots_thread(df, settings, plots=True, debug=False):
    """Stats quoted in paper which are not in the latex tables and plots

    Args:
        df (pandas.DataFrame) : summary statistics df
        settings (ExperimentSettings): custom class to hold hyperparameters
        plots (Boolean optional): make pltos or only printout stats 
        debug (Boolean optional): only print tasks
    Returns:
        printout: stats as organized in paper
        figures (png) : figures for paper at settings.dump_dir/figures/
        lightcurves (png): lightcurves used on paper at settings.dump_dir/lightcurves/modelname.*png
    """

    """
    Ordered as in paper
    """
    pd.set_option("max_colwidth", 1000)
    print(lu.str_to_greenstr(f"STATISTICS USED IN SUPERNNOVA"))

    # Baseline experiments
    baseline(df, settings, plots, debug)
    # Bayesian experiments
    # df_delta, df_delta_ood = sm.get_delta_metrics(df, settings)
    # bayesian(df, df_delta, df_delta_ood, settings, plots, debug)
    # Towards statistical analyses/cosmology
    # towards_cosmo(df, df_delta, df_delta_ood, settings, plots, debug)


def baseline(df, settings, plots, debug):
    """
    Baseline RNN
    """
    # 0. Figure example
    if plots:
        print(
            lu.str_to_yellowstr("Plotting candidates for Baseline binary (Figure 2.)")
        )
        model_file = f"{settings.models_dir}/{Base.replace('CNN_','CNN_S_0_')}/{Base.replace('CNN_','CNN_S_0_')}.pt"
        if os.path.exists(model_file):
            if debug:
                print(model_file)
            else:
                model_settings = conf.get_settings_from_dump(model_file)
                early_prediction.make_early_prediction(
                    model_settings, nb_lcs=20, do_gifs=True
                )
        else:
            print(lu.str_to_redstr(f"File not found {model_file}"))

    # 1. Hyper-parameters
    # saltfit, DF 0.2
    sel_criteria = ["CNN_CLF_2_R_None_saltfit_DF_0.2"]
    print(lu.str_to_bluestr(f"Hyperparameters {sel_criteria}"))
    if not debug:
        sm.get_metric_ranges(df, sel_criteria)

    # 3. Comparing with other methods
    print(lu.str_to_bluestr(f"Other methods:"))
    # Figure: accuracy vs. number of SNe
    if plots:
        print(lu.str_to_yellowstr("Plotting accuracy vs. SNe (Figure 3.)"))
        if not debug:
            sp.performance_plots(settings)
    # baseline best, saltfit
    if debug:
        print(Base_salt)
        print(Base_salt.replace("DF_1.0", "DF_0.05"))
        print(Base_salt.replace("DF_1.0", "DF_0.05").replace("None", "zpho"))
    else:
        sm.acc_auc_df(df, [Base_salt], data="saltfit")


    # 4. Redshift, contamination
    # baseline saltfit, 1.0, all redshifts
    sel_criteria = Base_salt.split("None")
    print("salt")
    if debug:
        print(sel_criteria, Base.split("None"))
    if not debug:
        sm.print_contamination(df, sel_criteria, settings, data="saltfit")
    print("photometry")
    if not debug:
        sm.print_contamination(df, Base.split("None"), settings, data="photometry")
