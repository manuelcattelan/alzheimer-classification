# Makefile

# project directories
DEFAULT_RAW_DIR = data/raw/
DEFAULT_PROCESSED_DIR = data/processed/
DEFAULT_RESULTS_DIR = results/
DEFAULT_SCRIPTS_DIR = src/

# either default or user-defined input path
ifndef INPUT
INPUT=$(dir ${wildcard ${DEFAULT_RAW_DIR}*/})
endif

# either default or user-defined output path
ifndef OUTPUT
OUTPUT=$(DEFAULT_RESULTS_DIR)
endif

PROCESSED_DATA = $(INPUT:$(DEFAULT_RAW_DIR)%=$(DEFAULT_PROCESSED_DIR)%)
RESULTS_DATA = $(PROCESSED_DATA:$(DEFAULT_PROCESSED_DIR)%=$(OUTPUT)%)

.PHONY: all clean $(PROCESSED_DATA) $(RESULTS_DATA)

all: $(PROCESSED_DATA) $(RESULTS_DATA)

$(PROCESSED_DATA):$(DEFAULT_PROCESSED_DIR)%:$(DEFAULT_RAW_DIR)%
	@if [ -f $< ]; then \
		python3 $(DEFAULT_SCRIPTS_DIR)data/build_data.py -f $<; \
	elif [ -d $< ]; then \
		python3 $(DEFAULT_SCRIPTS_DIR)data/build_data.py -d $<; \
	fi

$(RESULTS_DATA):$(OUTPUT)%:$(DEFAULT_PROCESSED_DIR)%
	@if [ -f $< ]; then \
		python3 $(DEFAULT_SCRIPTS_DIR)models/decision_tree.py -f $< -o $(OUTPUT); \
	elif [ -d $< ]; then \
		python3 $(DEFAULT_SCRIPTS_DIR)models/decision_tree.py -d $< -o $(OUTPUT); \
	fi

# clean intermediary (processed) files and results folder
clean:
	@echo "Cleaning $(DEFAULT_PROCESSED_DIR) folder..."
	@rm -rf $(DEFAULT_PROCESSED_DIR)
	@echo "Cleaning $(OUTPUT) folder..."
	@rm -rf $(OUTPUT)
	@echo "Done!"
