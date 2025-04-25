.PHONY: format lint clean run

all: format lint clean run

format: 
	black . --verbose
	isort . --profile black

isort:
	isort . --profile black

lint: 
	flake8 .

clean: 
	find . -type f -name "*.pyc" -delete 
	find . -type d -name "__pycache__" -delete

run: 
	python main.py
