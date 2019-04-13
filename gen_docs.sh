#!/usr/bin/env bash

PYTHONPATH=. python doc/gen_extend_pandas_docs.py |\
	rst2html.py --stylesheet-path=doc/style.css --template=doc/template.txt \
	> index.html

if [[ $1 = open ]] ; then
	chromium-browser index.html
fi
