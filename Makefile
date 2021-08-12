.PHONY: docs lint license format test FORCE

lint: FORCE
	flake8 scripts
	black --check scripts
	isort --check scripts

format: FORCE
	black scripts
	isort scripts

FORCE:
