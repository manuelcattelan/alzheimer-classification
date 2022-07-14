# makefile
ifndef input
input = data/raw
endif

ifndef processed
processed = data/processed
endif

ifndef output
output = reports
endif

ifndef models
models = $(notdir $(wildcard src/models/*.py))
endif

all: clean run

$(processed): $(input)
	python3 src/data/build_dataset.py -i $(input) -o $(processed)

dataset: $(processed)

run: dataset
	$(foreach model,$(models),@python3 src/models/$(model) -i $(processed) -o $(output)/$(basename $(model));)	
clean:
	rm -rf $(processed)
	rm -rf $(output)
