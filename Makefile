# Non-configurable paramters. Don't touch.
FILE := tfsenc_main
USR := $(shell whoami | head -c 2)
DT := $(shell date +"%Y%m%d-%H%M")

# -----------------------------------------------------------------------------
#  Configurable options
# -----------------------------------------------------------------------------

PRJCT_ID := podcast2
# {podcast | tfs}

# 625 Electrode IDs
SID := 625
# E_LIST := $(shell seq 1 1)
E_LIST := $(shell seq 1 105)

# # 676 Electrode IDs
SID := 676
E_LIST := $(shell seq 1 125)

PKL_IDENTIFIER := full-hs
# {full | trimmed}

# podcast electeode IDs
SID := 661
E_LIST :=  $(shell seq 1 115)
# SID := 662
# E_LIST :=  $(shell seq 1 100)
# SID := 717
# E_LIST :=  $(shell seq 1 255)
# SID := 723
# E_LIST :=  $(shell seq 1 165)
# SID := 741
# E_LIST :=  $(shell seq 1 130)
# SID := 742
# E_LIST :=  $(shell seq 1 175)
# SID := 743
# E_LIST :=  $(shell seq 1 125)
# SID := 763
# E_LIST :=  $(shell seq 1 80)
# SID := 798
# E_LIST :=  $(shell seq 1 195)
SID := 777
# number of permutations (goes with SH and PSH)
NPERM := 1000

# Choose the lags to run for.
LAGS := {-2000..2000..25}

CONVERSATION_IDX := 0

# Choose which set of embeddings to use
EMB := gpt2-xl
# {glove50 | gpt2-xl}
CNXT_LEN := 1024

# Choose the window size to average for each point
WS := 200

# Choose which set of embeddings to align with
#ALIGN_WITH := gpt2-xl
ALIGN_WITH := glove50
ALIGN_TGT_CNXT_LEN := 1024

# Specify the minimum word frequency
MWF := 1

# TODO: explain this parameter.
WV := all

# Choose whether to label or phase shuffle
# SH := --shuffle
# PSH := --phase-shuffle


# Choose whether to PCA the embeddings before regressing or not
PCA := --pca-flag
PCA_TO := 50

# num layers
LAYERS := {1..12..1}
# Choose the command to run: python runs locally, echo is for debugging, sbatch
# is for running on SLURM all lags in parallel.
#CMD := python
CMD := sbatch submit1.sh
# {echo | python | sbatch submit1.sh}

#TODO: move paths to makefile

# plotting modularity
# make separate models with separate electrodes (all at once is possible)
PDIR := $(shell dirname `pwd`)
link-data:
	ln -fs $(PDIR)/247-pickling/results/* data/

# -----------------------------------------------------------------------------
# Encoding
# -----------------------------------------------------------------------------
target1:
	for elec in $(E_LIST); do \
		$(CMD) code/$(FILE).py \
			--subject $(SID) \
			--lags $(LAGS) \
			--emb-file $(EMB) \
			--electrode $$elec \
			--npermutations 
			--output-folder $(SID)-$(USR)-test1; \
	done

# Run the encoding model for the given electrodes in one swoop
# Note that the code will add the subject, embedding type, and PCA details to
# the output folder name
run-encoding:
	mkdir -p logs
		$(CMD) code/$(FILE).py \
			--project-id $(PRJCT_ID) \
			--pkl-identifier $(PKL_IDENTIFIER) \
			--sid $(SID) \
			--conversation-id $(CONVERSATION_IDX) \
			--electrodes $(E_LIST) \
			--emb-type $(EMB) \
			--context-length $(CNXT_LEN) \
			--align-with $(ALIGN_WITH) \
			--align-target-context-length $(ALIGN_TGT_CNXT_LEN) \
			--window-size $(WS) \
			--word-value $(WV) \
			--npermutations $(NPERM) \
			--lags $(LAGS) \
			--min-word-freq $(MWF) \
			$(PCA) \
			--reduce-to $(PCA_TO) \
			$(SH) \
			$(PSH) \
			--output-parent-dir $(PRJCT_ID)-$(EMB)-reg-pca50d \
			--output-prefix '';\


run-sig-encoding:
	mkdir -p logs
		$(CMD) code/$(FILE).py \
			--project-id $(PRJCT_ID) \
			--pkl-identifier $(PKL_IDENTIFIER) \
			--conversation-id $(CONVERSATION_IDX) \
			--sig-elec-file bobbi.csv \
			--emb-type $(EMB) \
			--context-length $(CNXT_LEN) \
			--align-with $(ALIGN_WITH) \
			--align-target-context-length $(ALIGN_TGT_CNXT_LEN) \
			--window-size $(WS) \
			--word-value $(WV) \
			--npermutations $(NPERM) \
			--lags $(LAGS) \
			--min-word-freq $(MWF) \
			$(PCA) \
			--reduce-to $(PCA_TO) \
			$(SH) \
			$(PSH) \
			--output-parent-dir $(PRJCT_ID)-$(EMB)-pca50d-full-lm-out \
			--output-prefix '';\

#echo $$layer  \#
run-layered-sig-encoding:
	mkdir -p logs 
	for layer in $(LAYERS); do \
		$(CMD) code/$(FILE).py \
			--project-id $(PRJCT_ID) \
			--pkl-identifier $(PKL_IDENTIFIER)$$layer \
			--conversation-id $(CONVERSATION_IDX) \
			--sig-elec-file bobbi.csv \
			--emb-type $(EMB) \
			--context-length $(CNXT_LEN) \
			--align-with $(ALIGN_WITH) \
			--align-target-context-length $(ALIGN_TGT_CNXT_LEN) \
			--window-size $(WS) \
			--word-value $(WV) \
			--npermutations $(NPERM) \
			--lags $(LAGS) \
			--min-word-freq $(MWF) \
			$(PCA) \
			--reduce-to $(PCA_TO) \
			$(SH) \
			$(PSH) \
			--output-parent-dir $(PRJCT_ID)-$(EMB)-pca50d-full-hs$$layer \
			--output-prefix ''; \
	done

# Recommended naming convention for output_folder
#--output-prefix $(USR)-$(WS)ms-$(WV); \

# Run the encoding model for the given electrodes __one at a time__, ideally
# with slurm so it's all parallelized.
run-encoding-slurm:
	mkdir -p logs
	for elec in $(E_LIST); do \
		# for jobid in $(shell seq 1 1); do \
			$(CMD) code/$(FILE).py \
				--project-id $(PRJCT_ID) \
				--pkl-identifier $(PKL_IDENTIFIER) \
				--sig-elec-file bobbi.csv \
				--emb-type $(EMB) \
				--context-length $(CNXT_LEN) \
				--align-with $(ALIGN_WITH) \
				--align-target-context-length $(ALIGN_TGT_CNXT_LEN) \
				--window-size $(WS) \
				--word-value $(WV) \
				--npermutations $(NPERM) \
				--lags $(LAGS) \
				--min-word-freq $(MWF) \
				$(PCA) \
				--reduce-to $(PCA_TO) \
				$(SH) \
				$(PSH) \
				--output-parent-dir podcast-gpt2-xl-transcription \
				--output-prefix ''; \
				# --job-id $(EMB)-$$jobid; \
		# done; \
	done;


run-sig-encoding-slurm:
	mkdir -p logs
	for elec in $(E_LIST); do \
		# for jobid in $(shell seq 1 1); do \
			$(CMD) code/$(FILE).py \
				--project-id $(PRJCT_ID) \
				--pkl-identifier $(PKL_IDENTIFIER) \
				--sig-elec-file bobbi.csv \
				--emb-type $(EMB) \
				--context-length $(CNXT_LEN) \
				--align-with $(ALIGN_WITH) \
				--align-target-context-length $(ALIGN_TGT_CNXT_LEN) \
				--window-size $(WS) \
				--word-value $(WV) \
				--npermutations $(NPERM) \
				--lags $(LAGS) \
				--min-word-freq $(MWF) \
				$(PCA) \
				--reduce-to $(PCA_TO) \
				$(SH) \
				$(PSH) \
				--output-parent-dir podcast-gpt2-xl-transcription \
				--output-prefix ''; \
				# --job-id $(EMB)-$$jobid; \
		# done; \
	done;

pca-on-embedding:
	python code/tfsenc_pca.py \
			--sid $(SID) \
			--emb-type $(EMB) \
			--context-length $(CNXT_LEN) \
			--reduce-to $(EMB_RED_DIM); 

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
plot-encoding1:
	mkdir -p results/figures
	python code/tfsenc_plots.py \
			--sid $(SID) \
			--input-directory \
                                podcast-gpt2-xl-pca50d \
			--labels \
				gpt2-Fcnxt-pca50d \
			--output-file-name \
				'$(DT)-$(SID)-glove50_gpt2-xl_Fcnxt-gpt2-xl_Fcnxt-pca_50d-eric-true';
