.PHONY: setup run train test

setup:
	python3 setup_env.py --with-beatnet

run:
	python3 main.py

train:
	python3 train.py

test:
	python3 setup_env.py
	.venv/bin/python -m unittest discover -s tests -v
