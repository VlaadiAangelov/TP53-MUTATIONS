.PHONY: install download build eda binary multiclass interpret pipeline smoke clean

install:
	python3 -m pip install -e .

download:
	python3 scripts/00_download_data.py

build:
	python3 scripts/01_build_dataset.py

eda:
	python3 scripts/02_eda.py

binary:
	python3 scripts/03_train_binary.py

multiclass:
	python3 scripts/04_train_multiclass.py

interpret:
	python3 scripts/05_interpret.py

pipeline:
	python3 scripts/run_pipeline.py

smoke:
	python3 scripts/run_pipeline.py --skip-download --smoke-test

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

