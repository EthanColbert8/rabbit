#!/bin/bash
set -e

RECO="transformer"
STUDY_DIR="./templates/transformer2"

mkdir -p "$STUDY_DIR"

TIME=$(date "+%Y%m%d_%H%M%S")

python create_rabbit_input.py --outdir "$STUDY_DIR" --outname "rabbit_input.hdf5" --reco "$RECO" 2>&1 | tee "$STUDY_DIR/input_prep_$TIME.log"

rabbit_fit.py "$STUDY_DIR/rabbit_input.hdf5" -o "$STUDY_DIR" -t -1 --unblind \
    --allowNegativeParam --saveHists --saveHistsPerProcess --computeHistErrors --computeVariations \
    -m custom_mappings.steering.Steerability gen_emu gen_emu ttbar_dileptonic ttbar_dileptonic 'gen_cos_phiPrime:slice(6,12),gen_cos_phiPrime:sum' 'gen_cos_phiPrime:slice(0,6),gen_cos_phiPrime:sum' \
    --scan f_boosted --scanMappings Steerability \
    2>&1 | tee "$STUDY_DIR/fit_procedure_$TIME.log"

rabbit_plot_likelihood_scan.py "$STUDY_DIR/fitresults.hdf5" -o "$STUDY_DIR/fit_plots/" --param Steerability 2>&1 | tee "$STUDY_DIR/plotting_$TIME.log"
