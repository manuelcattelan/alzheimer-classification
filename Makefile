# Makefile

RAW_DATAPATH = data/raw
PROCESSED_DATAPATH = data/processed
RESULTS_DATAPATH = results
SCRIPTS_DATAPATH = src

RAW_CSV = $(wildcard $(RAW_DATAPATH)/*/*.csv)
PROCESSED_CSV = $(patsubst $(RAW_DATAPATH)/%.csv, $(PROCESSED_DATAPATH)/%.csv, $(RAW_CSV))

.PHONY: all clean

all: run

$(PROCESSED_CSV): $(PROCESSED_DATAPATH)/%.csv: $(RAW_DATAPATH)/%.csv $(SCRIPTS_DATAPATH)/data/build_data.py
	@python3 $(SCRIPTS_DATAPATH)/data/build_data.py -i $< -o $@

status:
	@echo "Cleansing data..."

run: status $(PROCESSED_CSV)
	@echo "Done!"
	@echo ""
	@cd $(SCRIPTS_DATAPATH)/models/ && python3 decision_tree_model.py

clean:
	rm -rf $(PROCESSED_DATAPATH)
	rm -rf $(RESULTS_DATAPATH)
