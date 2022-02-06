.PHONY: install lint format FORCE

install: FORCE
	pip install tapqir==1.0

lint: FORCE
	flake8 scripts
	black --check scripts
	isort --check scripts

format: FORCE
	black scripts
	isort scripts

analysis: FORCE
	python scripts/figures/DatasetA_ttfb_analysis.py
	python scripts/figures/DatasetB_ttfb_analysis.py
	python scripts/figures/DatasetC_ttfb_analysis.py
	python scripts/figures/DatasetD_ttfb_analysis.py

figures: FORCE
	python scripts/figures/graphical_model.py
	python scripts/figures/graphical_model_xy.py
	python scripts/figures/tapqir_analysis.py
	python scripts/figures/tapqir_analysis_probs.py
	python scripts/figures/tapqir_analysis_ppc.py
	python scripts/figures/tapqir_analysis_randomized.py
	python scripts/figures/tapqir_analysis_size.py
	python scripts/figures/tapqir_performance.py
	python scripts/figures/tapqir_performance_fn.py
	python scripts/figures/kinetic_analysis.py
	python scripts/figures/experimental_data.py
	python scripts/figures/experimental_data_DatasetA.py
	python scripts/figures/experimental_data_DatasetC.py
	python scripts/figures/experimental_data_DatasetD.py

supplementary: FORCE
	python scripts/supplementary/data1.py
	python scripts/supplementary/data2.py
	python scripts/supplementary/data3.py
	python scripts/supplementary/data4.py
	python scripts/supplementary/data5.py
	python scripts/supplementary/data6.py

FORCE:
