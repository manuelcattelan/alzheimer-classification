# Makefile

# project directories
DEFAULT_RAW_DIR = data/raw/
DEFAULT_PROCESSED_DIR = data/processed/
DEFAULT_RESULTS_DIR = results/
DEFAULT_SCRIPTS_DIR = src/

# either default or user-defined input-data path
ifndef INPUT
INPUT=$(dir ${wildcard ${DEFAULT_RAW_DIR}*/})
endif

# either default or user-defined processed-data path
ifndef PROCESSED
PROCESSED=$(DEFAULT_PROCESSED_DIR)
endif

# either default or user-defined output-data path
ifndef OUTPUT
OUTPUT=$(DEFAULT_RESULTS_DIR)
endif

ifndef MODELS
MODELS=$(wildcard $(DEFAULT_SCRIPTS_DIR)models/*.py)
endif

PROCESSED_DATA = $(INPUT:$(DEFAULT_RAW_DIR)%=$(PROCESSED)%)
RESULTS_DATA = $(PROCESSED_DATA:$(PROCESSED)%=$(OUTPUT)%)

.PHONY: all clean $(RESULTS_DATA)

all: $(PROCESSED_DATA) $(RESULTS_DATA)

$(PROCESSED_DATA):$(PROCESSED)%:$(DEFAULT_RAW_DIR)%
	@if [ -f $< ]; then \
		echo 'Executing $(DEFAULT_SCRIPTS_DIR)data/build_data.py -f $< -o $(PROCESSED)'; \
		python3 $(DEFAULT_SCRIPTS_DIR)data/build_data.py -f $< -o $(PROCESSED); \
	elif [ -d $< ]; then \
		echo 'Executing $(DEFAULT_SCRIPTS_DIR)data/build_data.py -d $< -o $(PROCESSED)'; \
		python3 $(DEFAULT_SCRIPTS_DIR)data/build_data.py -d $< -o $(PROCESSED); \
	fi

$(RESULTS_DATA):$(OUTPUT)%:$(PROCESSED)% $(MODELS)
	@for model in $(filter-out $<,$^); do \
		echo $$model; \
		if [ -f $< ]; then \
			python3 $$model -f $< -o $(PROCESSED); \
		elif [ -d $< ]; then \
			python3 $$model -d $< -o $(PROCESSED); \
		fi; \
	done

# clean intermediary (processed) files and results folder
clean:
	@echo "Cleaning $(PROCESSED) folder..."
	@rm -rf $(PROCESSED)
	@echo "Cleaning $(OUTPUT) folder..."
	@rm -rf $(OUTPUT)
	@echo "Done!"
