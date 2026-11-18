.PHONY: test test-collector test-classical test-synthetic test-shadow setup

ifeq ($(OS),Windows_NT)
  PY := .venv/Scripts/python.exe
else
  PY := .venv/bin/python
endif

setup:
	python scripts/bootstrap.py

test:
	python scripts/run_tests.py

test-collector:
	cd voice-dataset-collector && $(PY) -m unittest discover -s tests

test-classical:
	cd classical-anti-spoofing && $(PY) -m unittest discover -s tests

test-synthetic:
	cd synthetic-speech-dataset && $(PY) -m unittest discover -s tests

test-shadow:
	python -m unittest discover -s shadow_replay/tests -t .
