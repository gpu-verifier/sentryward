SHELL := /bin/sh
.DEFAULT_GOAL := build-cpp
BUILD_DIR ?= $(CURDIR)/build
CMAKE_BUILD_DIR ?= $(BUILD_DIR)/cmake
BOOTSTRAP_PYTHON ?= python3
PYTHON ?= $(BUILD_DIR)/venv/bin/python
CMAKE ?= $(if $(wildcard $(BUILD_DIR)/venv/bin/cmake),$(BUILD_DIR)/venv/bin/cmake,cmake)
BUILD_JOBS ?= 2
CMAKE_ARGS ?=
EXAMPLE_RUN_ARGS ?=
export PYTHONPYCACHEPREFIX := $(BUILD_DIR)/cache/python
export TMPDIR := $(BUILD_DIR)/tmp

.PHONY: prepare venv check-python check-cmake configure build-cpp rerun-examples rerun-examples-performance
prepare:
	mkdir -p "$(CMAKE_BUILD_DIR)" "$(BUILD_DIR)/artifacts" "$(BUILD_DIR)/cache/python" "$(BUILD_DIR)/tmp"
venv: prepare
	"$(BOOTSTRAP_PYTHON)" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ is required"'
	"$(BOOTSTRAP_PYTHON)" -m venv "$(BUILD_DIR)/venv"
check-python: prepare
	test -x "$(PYTHON)" || { echo "Run make venv or set PYTHON=/path/to/python3.10+." >&2; exit 1; }
	"$(PYTHON)" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ is required"'
check-cmake: prepare
	command -v "$(CMAKE)" >/dev/null 2>&1 || { echo "CMake 3.27+ is required; set CMAKE=/path/to/cmake." >&2; exit 1; }
configure: check-cmake
	"$(CMAKE)" -S . -B "$(CMAKE_BUILD_DIR)" -DSENTRYWARD_BUILD_INTERNAL_TOOLS=OFF $(CMAKE_ARGS)
build-cpp: configure
	"$(CMAKE)" --build "$(CMAKE_BUILD_DIR)" --parallel "$(BUILD_JOBS)"
rerun-examples: check-python build-cpp
	"$(PYTHON)" -m tools.rerun_examples $(EXAMPLE_RUN_ARGS)
rerun-examples-performance: check-python build-cpp
	"$(PYTHON)" -m tools.rerun_examples --include-timeouts $(EXAMPLE_RUN_ARGS)
