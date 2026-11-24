.PHONY: test audit check
PYTHON ?= python3
test:
	CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -B tools/run_cpu_tests.py
audit:
	$(PYTHON) -B -m robofate_pipeline audit --repo .
check: test audit
