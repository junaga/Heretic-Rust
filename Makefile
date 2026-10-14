.PHONY: build test setup setup-cpu setup-test setup-test-cpu parity

build:
	cargo build --release --locked

test:
	cargo test --locked

# Install the third-party framework runtime in an isolated environment.
setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install -r requirements.txt

# Install the minimal CPU framework runtime without upstream-oracle utilities.
setup-cpu:
	python3 -m venv .venv
	.venv/bin/python -m pip install -r requirements-runtime.lock --extra-index-url https://download.pytorch.org/whl/cpu

# Add the pinned upstream application's dependencies for differential tests.
setup-test:
	python3 -m venv .venv
	.venv/bin/python -m pip install -r requirements-test.txt

setup-test-cpu:
	python3 -m venv .venv
	.venv/bin/python -m pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu

# HERETIC_UPSTREAM must point to a checkout of the tag/commit in resources/upstream.json.
parity:
	test -n "$${HERETIC_UPSTREAM:-}"
	test "$$(git -C "$$HERETIC_UPSTREAM" rev-parse HEAD)" = "6ea3b8d778d047b4b3b7c5b843e21c5bea98ee8d"
	PYTHONPATH="$$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" OMP_NUM_THREADS="1" NUMBA_NUM_THREADS="1" MPLBACKEND="Agg" cargo test --locked -- --include-ignored --test-threads=1
