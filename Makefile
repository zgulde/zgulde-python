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

gh-pages: clean ## Build, commit, and push docs for the github-pages branch
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

test: ## Run the tests for zgulde/extend_pandas
	python -m doctest zgulde/extend_pandas.py

