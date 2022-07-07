# Makefile

# Environmental variables from commandline
# default input data is the entire raw folder
# default models to run are all
input_path?=${dir ${wildcard data/raw/*/}}
models?=all

# project directories
raw_dir=data/raw
processed_dir=data/processed
results_dir=results
scripts_dir=src

input=$(input_path)
processed=$(patsubst $(raw_dir)%,$(processed_dir)%,$(input))
results=$(patsubst $(processed_dir)%,$(results_dir)%,$(processed))

$(info $(input))
$(info $(processed))
$(info $(results))

.PHONY: all $(processed) $(results)

all: $(processed) $(results) 

$(results): $(results_dir)%: $(processed_dir)% $(scripts_dir)/models/decision_tree.py
	python3 $(scripts_dir)/models/decision_tree.py -i $< -o $@

$(processed): $(processed_dir)%: $(raw_dir)% $(scripts_dir)/data/build_data.py
	python3 $(scripts_dir)/data/build_data.py -i $< -o $@

clean:
	@echo "Cleaning $(processed_dir)/ folder..."
	@rm -rf $(processed_dir)
	@echo "Cleaning $(results_dir)/ folder..."
	@rm -rf $(results_dir)
	@echo "Done!"
