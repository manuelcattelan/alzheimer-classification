# Makefile

.PHONY: all clean data

all: run_dt

# Target for processing data from raw to processed
data: data/raw/air data/raw/paper data/raw/ap src/data/build_data.py
	@echo ""
	@echo "Running src/data/build_data.py..."
	@cd src/data/ && python3 build_data.py 

# Target for running decision tree classification on processed data
run_dt: data src/models/decision_tree_model.py
	@echo ""
	@echo "Running src/models/decision_tree_model.py..."
	@cd src/models && python3 decision_tree_model.py

# Clean target
clean:
	@rm -rf data/processed/
	@rm -rf results/
