default: test-extend-pandas lint

.PHONY: clean
clean: ## Remove built docs and packaging artifacts
	rm -rf dist build zgulde.egg-info public
	rm -f index.html

.PHONY: release
release: clean test lint gh-pages ## Release a new version to pypi
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

.PHONY: test test-em test-extend-pandas test-util
test-em: ## Run the tests for the `em` module
	pytest -q zgulde/em/test_extract_markdown.py
test-extend-pandas: ## Run the doctests for the zgulde/extend_pandas module
	python -m doctest -o NORMALIZE_WHITESPACE -o ELLIPSIS zgulde/extend_pandas.py
test-util: ## Run the tests for zgulde/__init__
	python -m doctest -o NORMALIZE_WHITESPACE -o ELLIPSIS zgulde/__init__.py
	pytest -q zgulde/test_utility_functions.py
test-flashcards:
	pytest -q zgulde/flashcards/test_flashcards.py
	pytype zgulde/flashcards/__main__.py
test-my-range:
	pytest -q --disable-warnings zgulde/test_my_range.py
test: test-em test-extend-pandas test-util test-my-range test-flashcards ## Run all the tests

.PHONY: lint-pytype lint-mypy lint
lint-pytype:
	pytype zgulde/extend_pandas.py
	pytype zgulde/__init__.py
lint-mypy:
	python -m mypy zgulde/__init__.py
lint: lint-pytype lint-mypy ## Check types

.PHONY: help
help: ## Show this help message
	@grep -E '^[a-zA-Z._-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%s\033[0m\t%s\n", $$1, $$2}' | column -ts$$'\t'
