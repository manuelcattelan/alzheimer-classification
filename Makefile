# Makefile

ENV_NAME = alzheimer-thesis

raw ?= data/raw
processed ?= $(subst raw,processed,$(raw))
reports ?= reports

create_environment:
	conda env create --name $(ENV_NAME) --file environment.yml

update_environment:
	conda env update --name $(ENV_NAME) --file environment.yml

dataset:
	python3 src/data/make_dataset.py --input $(raw) --output $(processed)

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf $(processed)
	rm -rf $(reports)
