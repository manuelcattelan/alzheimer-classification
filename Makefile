# makefile
default_input = data/raw
input ?= $(default_input)
processed ?= $(subst raw,processed,$(input))
output ?= reports
models ?= $(notdir $(wildcard src/models/*.py))

$(info $(input))
$(info $(processed))
$(info $(output))

input_root = $(subst /, ,$(default_input))
input_custom = $(subst /, ,$(input))

$(info $(input_root))
$(info $(input_custom))

empty=
space= $(empty) $(empty)

input_pattern = $(filter-out $(input_root), $(input_custom))
subst_input_pattern = $(patsubst %.csv,%.png,$(input_pattern))

new_input_pattern = $(subst $(space),/,$(subst_input_pattern))

$(info $(input_pattern))
$(info $(new_input_pattern))

all: clean run

$(processed): $(input)
	python3 src/data/build_dataset.py -i $(input) -o $(processed)

dataset: $(processed)

run: dataset
	$(foreach model,$(models),python3 src/models/$(model) -i $(processed) -o $(output)/$(basename $(model))/$(new_input_pattern);)	

clean:
	rm -rf $(processed)
	rm -rf $(output)
