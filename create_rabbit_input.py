import os
import re
import logging
import argparse
import numpy as np
import hist
from hist import Hist
from coffea import util as coffea_utils
from rabbit import tensorwriter

hist_file_patterns = {
    "ee": re.compile(r"Histogram_ee_(.*)\.coffea"),
    "emu": re.compile(r"Histogram_emu_(.*)\.coffea"),
    "mumu": re.compile(r"Histogram_mumu_(.*)\.coffea"),
    "combined": re.compile(r"Histogram_combined_(.*)\.coffea"),
}

signal_processes = ["ttbar_dileptonic"] # add etat_dileptonic later, or maybe it's a background...

def load_histograms(histogram_dir, files_list, channel="combined"):
    histograms = {}
    for filename in files_list:
        process_name = hist_file_patterns[channel].fullmatch(filename).group(1)
        histograms[process_name] = coffea_utils.load(os.path.join(histogram_dir, filename))
    return histograms

def process_is_signal(process_name):
    return process_name in signal_processes

def apply_boosted_cuts(histogram, prefix="transformer_"):
    ttbar_mass = [800.0, 1500.0]
    cos_theta = [-0.4, 0.4]
    return histogram[
        hist.loc(ttbar_mass[0]):hist.loc(ttbar_mass[1]):sum, # ttbar mass
        hist.loc(cos_theta[0]):hist.loc(cos_theta[1]):sum, # cos(theta)
        ::sum, # beta
        ...
    ].project(f"{prefix}cos_phiPrime")

def get_templates_folder(reco="hybrid", nosc=False):
    match reco.lower():
        case "hybrid": return "4_vars_14_5_7_13hybrid2nosc-0" if nosc else "4_vars_14_5_7_13hybrid2-0"
        case "mlb": return "4_vars_14_5_7_13mlb2nosc-0" if nosc else "4_vars_14_5_7_13mlb2-0"
        case "transformer": return "4_vars_14_5_7_13transformer2nosc-0" if nosc else "4_vars_14_5_7_13transformer2-0"
        case "gen": return "4_varsgen2nosc-0" if nosc else "4_varsgen2-0"
    raise ValueError(f"Reconstruction \"{reco}\" not recognized (must be \"hybrid\", \"transformer\", \"mlb\", or \"gen\")")

def main(args):
    hist_templates_dir = "/depot/cms/top/colberte/top-spincorr-framework/histogram_templates"
    channels = ["ee", "emu", "mumu"]
    normal_hist_prefix = "" if args.reco.lower()=="mlb" else f"{args.reco}_"

    save_folder = args.outdir if (args.outdir is not None) else os.path.join(os.path.dirname(hist_templates_dir), "rabbit_templates")
    os.makedirs(save_folder, exist_ok=True)

    nominal_hists_dir = os.path.join(hist_templates_dir, get_templates_folder(args.reco, nosc=False), "Nominal")
    nosc_hists_dir = os.path.join(hist_templates_dir, get_templates_folder(args.reco, nosc=True), "noSpinCorr")

    gen_nominal_hists_dir = os.path.join(hist_templates_dir, get_templates_folder("gen", nosc=False), "Nominal")
    gen_nosc_hists_dir = os.path.join(hist_templates_dir, get_templates_folder("gen", nosc=True), "noSpinCorr")

    writer = tensorwriter.TensorWriter(systematic_type="normal")

    logging.info(f"Preparing channels {channels}")
    for channel in channels:
        nominal_files = list(filter(hist_file_patterns[channel].search, os.listdir(nominal_hists_dir)))
        nosc_files = list(filter(hist_file_patterns[channel].search, os.listdir(nosc_hists_dir)))

        gen_nominal_files = list(filter(hist_file_patterns[channel].search, os.listdir(gen_nominal_hists_dir)))
        gen_nosc_files = list(filter(hist_file_patterns[channel].search, os.listdir(gen_nosc_hists_dir)))

        logging.info(f"Found {len(nominal_files)} nominal processes and {len(nosc_files)} no-spincorr processes for channel \"{channel}\".")

        nominal_hists = load_histograms(nominal_hists_dir, nominal_files, channel=channel)
        nosc_hists = load_histograms(nosc_hists_dir, nosc_files, channel=channel)

        gen_nominal_hists = load_histograms(gen_nominal_hists_dir, gen_nominal_files, channel=channel)
        gen_nosc_hists = load_histograms(gen_nosc_hists_dir, gen_nosc_files, channel=channel)

        axes = apply_boosted_cuts(nominal_hists[signal_processes[0]]["Nominal", ...], prefix=normal_hist_prefix).axes
        writer.add_channel(axes, name=channel)

        for p, h in nominal_hists.items():
            the_hist = apply_boosted_cuts(h["Nominal", ...], prefix=normal_hist_prefix)

            logging.debug(f"Adding nominal process \"{p}\".")
            writer.add_process(the_hist, name=p, channel=channel, signal=process_is_signal(p))

        for p in signal_processes:
            h = nosc_hists.get(p, None)
            if (h is None):
                logging.warning(f"Signal process \"{p}\" does has no noSpinCorr variation. Skipping systematic addition.")
                continue

            logging.info(f"Adding noSpinCorr systematic for signal process \"{p}\".")

            the_hist = apply_boosted_cuts(h["noSpinCorr", ...], prefix=normal_hist_prefix)
            nominal_hist = apply_boosted_cuts(nominal_hists[p]["Nominal", ...], prefix=normal_hist_prefix)

            nosc_up_hist = (0.8 * nominal_hist) + (0.2 * the_hist)
            nosc_down_hist = (2.0 * nominal_hist) + (-1.0 * nosc_up_hist) # mirror up variation across nominal

            writer.add_systematic(
                [nosc_up_hist, nosc_down_hist], "f_boosted", p, channel,
                symmetrize="average", constrained=False, noi=True
            )

        gen_axes = apply_boosted_cuts(gen_nominal_hists[signal_processes[0]]["Nominal", ...], prefix="gen_").axes
        writer.add_channel(gen_axes, name=f"gen_{channel}", masked=True)

        for p, h in gen_nominal_hists.items():
            the_hist = apply_boosted_cuts(h["Nominal", ...], prefix="gen_")

            logging.debug(f"Adding nominal process \"{p}\" to masked gen channel.")
            writer.add_process(the_hist, name=p, channel=f"gen_{channel}", signal=process_is_signal(p))

        for p in signal_processes:
            h = gen_nosc_hists.get(p, None)
            if (h is None):
                logging.warning(f"Signal process \"{p}\" does has no noSpinCorr variation. Skipping systematic addition.")
                continue

            logging.info(f"Adding noSpinCorr systematic for signal process \"{p}\" to masked gen channel.")

            the_hist = apply_boosted_cuts(h["noSpinCorr", ...], prefix="gen_")
            nominal_hist = apply_boosted_cuts(gen_nominal_hists[p]["Nominal", ...], prefix="gen_")

            nosc_up_hist = (0.8 * nominal_hist) + (0.2 * the_hist)
            nosc_down_hist = (2.0 * nominal_hist) + (-1.0 * nosc_up_hist) # mirror up variation across nominal

            writer.add_systematic(
                [nosc_up_hist, nosc_down_hist], "f_boosted", p, f"gen_{channel}",
                symmetrize="average", constrained=False, noi=True
            )

        # logging.info("Adding norm systematic for all processes.")
        # all_processes = [*nominal_hists.keys(), *[f"noSpinCorr_{key}" for key in nosc_hists.keys()]]
        # for p in all_processes:
        #     writer.add_norm_systematic(f"{p}_norm", p, "combined", 1.05)

        logging.info(f"Adding fake data taken from (0.98*nominal + 0.02*nosc) for channel \"{channel}\".")
        fake_data_hist = (0.98 * apply_boosted_cuts(nominal_hists[signal_processes[0]]["Nominal", ...], prefix=normal_hist_prefix)) + (0.02 * apply_boosted_cuts(nosc_hists[signal_processes[0]]["noSpinCorr", ...], prefix=normal_hist_prefix))
        for p in nominal_hists.keys():
            if (p == signal_processes[0]):
                continue
            fake_data_hist += apply_boosted_cuts(nominal_hists[p]["Nominal", ...], prefix=normal_hist_prefix)
        writer.add_data(fake_data_hist, channel)

    writer.write(save_folder, args.outname)
    logging.info(f"Rabbit inputs written to \"{save_folder}/{args.outname}\".")

if (__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default=None, help="Output directory. Default is \"./rabbit_templates\"")
    parser.add_argument("--outname", type=str, default="rabbit_input_nosc_combined.hdf5", help="Output filename. Default is \"rabbit_input_nosc_combined.hdf5\"")
    parser.add_argument("--reco", type=str, default="hybrid", help="Name of reconstruction method to get templates from. Default is \"hybrid\"")
    parser.add_argument("--verbose", action="store_true", help="Enable debug-level logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="[%(asctime)s - %(module)s - %(levelname)s]: %(message)s")
    logging.getLogger("fsspec.local").setLevel(logging.INFO) # suppress logs for every file access
    logging.getLogger("boost_histogram.histogram").setLevel(logging.INFO) # suppress logs for every histogram operation
    logging.getLogger("h5py._conv").setLevel(logging.INFO) # suppress confusing logs for h5py conversions

    logging.info("Beginning preparation of rabbit input tensor.")
    main(args)
    logging.info("Finished.")
