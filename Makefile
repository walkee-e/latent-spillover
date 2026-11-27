PYTHON ?= python3

.PHONY: verify figures test plan
verify:
	$(PYTHON) scripts/verify_results.py

figures: verify
	$(PYTHON) scripts/reproduce_figures.py

test: verify
	$(PYTHON) -m unittest discover -s experiments/optimizer_grid -p 'test_*.py'
	cd 3_2_german_city_names/rank_sweep && $(PYTHON) -m unittest test_experiment
	cd 4_1_israeli_dishes/rank_sweep && $(PYTHON) -m unittest test_experiment
	cd 5_2_evil_terminator/rank_sweep && $(PYTHON) -m unittest test_experiment

plan:
	$(PYTHON) experiments/optimizer_grid/run.py --dry-run
