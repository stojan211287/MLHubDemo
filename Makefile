.PHONY: venv format lint start

build: requirements.txt
	./venv/bin/pip-sync requirements.txt

dev: requirements-dev.txt
	./venv/bin/pip-sync requirements-dev.txt

format:
	./venv/bin/black ./mlhub

requirements.txt: requirements.in venv
	./venv/bin/pip-compile requirements.in --output-file=requirements.txt

requirements-dev.txt: requirements-dev.in venv
	./venv/bin/pip-compile requirements-dev.in --output-file=requirements-dev.txt

lint:
	./venv/bin/flake8 ./mlhub
start:
	export PYTHONPATH="./mlhub"
	./venv/bin/python3 ./mlhub/app.py

venv:
	python3 -m venv venv
	./venv/bin/pip3 install --upgrade pip
	./venv/bin/pip3 install pip-tools
