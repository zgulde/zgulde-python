.PHONY: default docs release clean help

default: help

clean: ## Remove built docs and packaging artifacts
	rm -rf dist build zgulde.egg-info
	rm -f index.html

release: clean ## Release a new version to pypi
	python3 setup.py sdist bdist_wheel
	python3 -m twine upload dist/*

docs: ## Build the docs for extend_pandas
	PYTHONPATH=. python doc/gen_extend_pandas_docs.py |\
		rst2html.py --stylesheet-path=doc/style.css \
		--template=doc/template.txt \
		> index.html

help: ## Show this help message
	@grep -E '^[a-zA-Z._-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[34m%s\033[0m\t%s\n", $$1, $$2}' | column -ts$$'\t'
