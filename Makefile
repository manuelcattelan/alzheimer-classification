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
models = $(wildcard src/models/*.py)
endif

data/processed: data/raw
	python3 src/data/build_dataset.py -i $(input) -o $(processed)

run: data/processed
	@for model in $(models); do \
		python3 $$model -i $(processed) -o $(output); \
	done

clean:
	rm -rf $(processed)
	rm -rf $(output)
