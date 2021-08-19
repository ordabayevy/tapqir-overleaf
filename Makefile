.PHONY: docs lint license format test FORCE

lint: FORCE
	flake8 scripts
	black --check scripts
	isort --check scripts

format: FORCE
	black scripts
	isort scripts

analysis: FORCE
	python scripts/figures/analysis7.py
	python scripts/extended-data/analysis4.py
	python scripts/extended-data/analysis5.py
	python scripts/extended-data/analysis6.py

figures: FORCE
	python scripts/figures/figure2.py
	python scripts/figures/figure3.py
	python scripts/figures/figure4.py
	python scripts/figures/figure5.py
	python scripts/figures/figure6.py
	python scripts/figures/figure7.py

extended-data: FORCE
	python scripts/extended-data/figure1.py
	python scripts/extended-data/figure2.py
	python scripts/extended-data/figure3.py
	python scripts/extended-data/figure4.py
	python scripts/extended-data/figure5.py
	python scripts/extended-data/figure6.py

supplementary: FORCE
	python scripts/supplementary/data1.py
	python scripts/supplementary/data2.py
	python scripts/supplementary/data3.py
	python scripts/supplementary/data4.py
	python scripts/supplementary/data5.py

FORCE:
