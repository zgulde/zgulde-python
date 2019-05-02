.PHONY: default docs release clean help test gh-pages

default: help

clean: ## Remove built docs and packaging artifacts
	rm -rf dist build zgulde.egg-info public
	rm -f index.html

release: clean test gh-pages ## Release a new version to pypi
	python3 setup.py sdist bdist_wheel
	python3 -m twine upload dist/*

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

test: ## Run the tests for zgulde/extend_pandas and zgulde/__init__
	python -m doctest -o NORMALIZE_WHITESPACE -o ELLIPSIS zgulde/extend_pandas.py
	python -m doctest -o NORMALIZE_WHITESPACE -o ELLIPSIS zgulde/__init__.py

help: ## Show this help message
	@grep -E '^[a-zA-Z._-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%s\033[0m\t%s\n", $$1, $$2}' | column -ts$$'\t'
