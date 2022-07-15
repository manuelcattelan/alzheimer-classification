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
default_processed_dir_list = $(subst /, ,$(default_processed_data))
custom_processed_dir_list = $(subst /, ,$(processed_data))

custom_processed_only_dir_list = $(filter-out $(default_processed_dir_list), $(custom_processed_dir_list))
final_custom_processed_only_dir_list = $(patsubst %.csv,%.png,$(custom_processed_only_dir_list))

custom_output_path = $(subst $(space),/,$(final_custom_processed_only_dir_list))

all: clean run
	$(info Classification has finished, results can be found in: $(output_data))

$(processed_data): $(input_data)
	$(info Building $(input_data)...)
	@python3 src/data/build_dataset.py -i $(input_data) -o $(processed_data)

dataset: $(processed_data)
	$(info $(input_data) was built, processed data can be found in: $(processed_data))

run: dataset
	@$(foreach model,$(models), \
		python3 src/models/$(model) -i $(processed_data) -o $(output_data)/$(basename $(model))/$(custom_output_path); \
	)

clean:
	$(info Removing $(processed_data) if present...)
	@rm -rf $(processed_data)
	$(info Removing $(output_data) if present...)
	@rm -rf $(output_data)
	$(info Done!)
