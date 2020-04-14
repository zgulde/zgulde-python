default: doctest

.PHONY: pre-commit
pre-commit: fmt test check-types

.PHONY: clean
clean: ## Remove built docs and packaging artifacts
	rm -rf dist build zgulde.egg-info public
	rm -f index.html
	rm -rf .make .mypy_cache .pytest_cache .pytype

.PHONY: release
release: fmt test check-types gh-pages ## Release a new version to pypi
	python3 setup.py sdist bdist_wheel
	python3 -m twine upload dist/*

.PHONY: docs gh-pages
docs: ## Build the docs for extend_pandas
	mkdir -p public
	PYTHONPATH=. python doc/gen_extend_pandas_docs.py |\
		rst2html.py --stylesheet-path=doc/style.css \
		--template=doc/template.txt \
		> public/index.html
gh-pages: clean ## Build, commit, and push docs for the gh-pages branch
	mkdir -p public
	@if [[ ! -f .git/refs/heads/gh-pages ]] ; then \
		@echo '[make] Creating gh-pages branch';\
		git --work-tree public checkout -q --orphan gh-pages;\
		touch public/index.html;\
		git --work-tree public add --all;\
		git --work-tree public commit --quiet --message 'Update Docs';\
		git checkout --quiet --force master;\
	fi
	git --work-tree public checkout gh-pages
	make docs
	git --work-tree public add -A
	git --work-tree public commit --amend --no-edit
	git checkout --force master
	git push origin gh-pages --force

PY_FILES := $(shell find zgulde -name \*.py)

.PHONY: test doctest pytest
DOCTESTS := $(addprefix .make/doctest/, $(filter-out %__main__.py, $(PY_FILES)))
.make/doctest/%: %
	@mkdir -p $(dir $@)
	python -m doctest -o NORMALIZE_WHITESPACE -o ELLIPSIS $<
	@touch $@
test: doctest pytest
doctest: $(DOCTESTS)
pytest:
	python -m pytest

.PHONY: check-types check-types-mypy check-types-pytype
PYTYPE_TYPE_CHECKS := $(addprefix .make/pytype/, $(PY_FILES))
MYPY_TYPE_CHECKS := $(addprefix .make/mypy/, $(PY_FILES))
.make/pytype/%: %
	@mkdir -p $(dir $@)
	pytype $<
	@touch $@
.make/mypy/%: %
	@mkdir -p $(dir $@)
	python -m mypy --ignore-missing-imports $<
	@touch $@
check-types-mypy: $(MYPY_TYPE_CHECKS)
check-types-pytype: $(PYTYPE_TYPE_CHECKS)
check-types: check-types-mypy check-types-pytype

.PHONY: fmt
fmt: ## Format code with isort and black
	python -m isort --line-width 88 --trailing-comma --multi-line 3 $(PY_FILES)
	python -m black -q $(PY_FILES)

.PHONY: help
help: ## Show this help message
	@grep -E '^[a-zA-Z._-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%s\033[0m\t%s\n", $$1, $$2}' | column -ts$$'\t'
