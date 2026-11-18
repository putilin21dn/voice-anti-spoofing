.PHONY: test test-collector test-classical test-synthetic

test: test-collector test-classical test-synthetic

test-collector:
	cd voice-dataset-collector && python -m unittest discover -s tests

test-classical:
	cd classical-anti-spoofing && python -m unittest discover -s tests

test-synthetic:
	cd synthetic-speech-dataset && python -m unittest discover -s tests
