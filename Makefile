.PHONY: default docs release clean help test

default: help

clean: ## Remove built docs and packaging artifacts
	rm -rf dist build zgulde.egg-info
	rm -f index.html

release: clean test docs ## Release a new version to pypi
	python3 setup.py sdist bdist_wheel
	python3 -m twine upload dist/*
	myserver upload --file index.html --destination /srv/zach.lol/public/extend_pandas.html

docs: ## Build the docs for extend_pandas
	PYTHONPATH=. python doc/gen_extend_pandas_docs.py |\
		rst2html.py --stylesheet-path=doc/style.css \
		--template=doc/template.txt \
		> index.html

test: ## Run the tests for zgulde/extend_pandas
	python -m doctest zgulde/extend_pandas.py

help: ## Show this help message
	@grep -E '^[a-zA-Z._-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%s\033[0m\t%s\n", $$1, $$2}' | column -ts$$'\t'
