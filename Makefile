.PHONY: build test setup setup-cpu parity

build:
	cargo build --release --locked

test:
	cargo test --locked

# Install the third-party framework runtime in an isolated environment.
setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install -r requirements.txt

# Reproduce the tested CPU environment, including optional research dependencies.
setup-cpu:
	python3 -m venv .venv
	.venv/bin/python -m pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu

# HERETIC_UPSTREAM must point to a checkout of the tag/commit in resources/upstream.json.
parity:
	test -n "$${HERETIC_UPSTREAM:-}"
	PYTHONPATH="$$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" OMP_NUM_THREADS="1" NUMBA_NUM_THREADS="1" MPLBACKEND="Agg" cargo test --locked -- --include-ignored --test-threads=1
