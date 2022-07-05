# Makefile
input_path?=${dir ${wildcard data/raw/*/}}

raw_dir=data/raw
processed_dir=data/processed
results_dir=results
scripts_dir=src

input_data=$(wildcard $(addsuffix *csv, $(basename $(input_path))))
output_data=$(patsubst $(raw_dir)%.csv, $(processed_dir)%.csv, $(input_data))

.PHONY: data all

all: run-all

$(output_data): $(processed_dir)%.csv: $(raw_dir)%.csv $(scripts_dir)/data/build_data.py
	python3 $(scripts_dir)/data/build_data.py -i $< -o $@

run-single: data
	@cd $(scripts_dir)/models/ && python3 decision_tree_model.py

run-folder: data
	@cd $(scripts_dir)/models/ && python3 decision_tree_model.py

run-all: data
	@cd $(scripts_dir)/models/ && python3 decision_tree_model.py

data: $(output_data)

clean:
	@echo "Cleaning $(processed_dir)/ folder..."
	@rm -rf $(processed_dir)
	@echo "Cleaning $(results_dir)/ folder..."
	@rm -rf $(results_dir)
	@echo "Done!"
