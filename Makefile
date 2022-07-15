# makefile

# from manual, used for whitespace -> '/' substitution
empty=
space= $(empty) $(empty)

# default data directories
default_input_data = data/raw
default_processed_data = data/processed
default_output_data = reports

# custom data directories
input_data ?= $(default_input_data)
processed_data ?= $(default_processed_data)
output_data ?= $(default_output_data)
models ?= $(notdir $(wildcard src/models/*.py))

# input path substitution to build full output path
default_input_dir_list = $(subst /, ,$(default_input_data))
custom_input_dir_list = $(subst /, ,$(input_data))

custom_input_only_dir_list = $(filter-out $(default_input_dir_list), $(custom_input_dir_list))
final_custom_input_only_dir_list = $(patsubst %.csv,%.png,$(custom_input_only_dir_list))

custom_output_path = $(subst $(space),/,$(final_custom_input_only_dir_list))

all: clean run

$(processed_data): $(input_data)
	python3 src/data/build_dataset.py -i $(input_data) -o $(processed_data)

dataset: $(processed_data)

run: dataset
	$(foreach model,$(models),python3 src/models/$(model) -i $(processed_data) -o $(output_data)/$(basename $(model))/$(custom_output_path);)

clean:
	rm -rf $(processed_data)
	rm -rf $(output_data)
