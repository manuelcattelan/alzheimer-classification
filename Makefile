# Makefile

# Environmental variables from commandline
# default input data is the entire raw folder
# default models to run are all
input?=${dir ${wildcard data/raw/*/}}
models?=all

# project directories
raw_dir=data/raw/
processed_dir=data/processed/
results_dir=results/
scripts_dir=src/

# raw input data name (either file or directory)
input_data=$(input)
# preprocessed input data name (either file or folder)
preprocessed_data=$(input_data:$(raw_dir)%=$(processed_dir)%)
# results input data name (either file or folder)
results_data=$(preprocessed_data:$(processed_dir)%=$(results_dir)%)

# create directory for processed data
$(shell mkdir -p ${dir ${preprocessed_data}})
# create directory for results data
$(shell mkdir -p ${dir ${results_data}})

.PHONY: all $(preprocessed_data) $(results_data)

# build processed data and run classification on it
all: $(preprocessed_data) $(results_data) 

# run preprocessing on raw data and store results in processed directory
$(preprocessed_data): $(processed_dir)%:$(raw_dir)% $(scripts_dir)data/build_data.py
	@python3 $(scripts_dir)data/build_data.py -i $< -o $@

# run classification on processed data and store results in results directory
$(results_data): $(results_dir)%:$(processed_dir)% $(scripts_dir)models/decision_tree.py
	@python3 $(scripts_dir)models/decision_tree.py -i $< -o $@

# clean intermediary (processed) files and results folder
clean:
	@echo "Cleaning $(processed_dir) folder..."
	@rm -rf $(processed_dir)
	@echo "Cleaning $(results_dir) folder..."
	@rm -rf $(results_dir)
	@echo "Done!"
