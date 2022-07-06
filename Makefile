# Makefile

# Environmental variables from commandline
# default input data is the entire raw folder
# default models to run are all
input_path ?= ${dir ${wildcard data/raw/*/}}
models ?= all

# project directories
raw_dir=data/raw
processed_dir=data/processed
results_dir=results
scripts_dir=src

# all .csv files from input path
input_data=$(wildcard $(addsuffix *csv, $(basename $(input_path))))
# all .csv files to export as output
processed_data=$(patsubst $(raw_dir)%.csv, $(processed_dir)%.csv, $(input_data))
# all .txt files to which classification results are stored
results_data=$(patsubst $(processed_dir)%.csv, $(results_dir)%.txt, $(processed_data))

.PHONY: all $(input_path)

all: $(input_path)

$(input_path):
	@if [ -f $@ ]; then \
		echo 'file'; \
	elif [ -d $@ ]; then \
		echo 'dir'; \
	fi

$(results_data): $(results_dir)%.txt: $(processed_dir)%.csv $(scripts_dir)/models/decision_tree.py
	python3 $(scripts_dir)/models/decision_tree.py -i $< -o $@

$(processed_data): $(processed_dir)%.csv: $(raw_dir)%.csv $(scripts_dir)/data/build_data.py
	python3 $(scripts_dir)/data/build_data.py -i $< -o $@

clean:
	@echo "Cleaning $(processed_dir)/ folder..."
	@rm -rf $(processed_dir)
	@echo "Cleaning $(results_dir)/ folder..."
	@rm -rf $(results_dir)
	@echo "Done!"
