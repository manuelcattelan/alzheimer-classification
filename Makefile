# Makefile

ENV_NAME = alzheimer-thesis
ENV_CONFIG = environment.yml

DEFAULT_RAW_DATA = data/raw
DEFAULT_PROCESSED_DATA = data/processed

raw ?= $(DEFAULT_RAW_DATA)
processed ?= $(DEFAULT_PROCESSED_DATA)

create_environment:
	conda env create --name $(ENV_NAME) --file $(ENV_CONFIG)

update_environment:
	conda env update --name $(ENV_NAME) --file $(ENV_CONFIG)

dataset:
	python3 src/data/make_dataset.py --input $(raw) --output $(processed)

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf $(processed)
	rm -rf $(reports)
