.PHONY: setup run train train-cnn train-gaussian evaluate evaluate-rwc test

setup:
	python3 setup_env.py --with-beatnet

run:
	.venv/bin/python main.py

train:
	.venv/bin/python train_hierarchical_cnn.py

train-cnn:
	.venv/bin/python train_hierarchical_cnn.py

train-gaussian:
	.venv/bin/python train_gaussian_baseline.py

evaluate:
	.venv/bin/python evaluate_hierarchical.py

evaluate-rwc:
	.venv/bin/python evaluate_rwc.py

test:
	python3 setup_env.py
	.venv/bin/python -m unittest discover -s tests -v
