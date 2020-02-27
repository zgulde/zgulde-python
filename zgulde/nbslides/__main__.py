import json
import os
import sys


def display_markdown(cell, output_slide_seperator=True, classes=None):
    if output_slide_seperator:
        print("---\n")
    if classes is not None:
        print("class: " + ", ".join(classes) + "\n")
    print("".join(cell["source"]))


def display_code(cell, show_source=True, show_execute_results=True):
    # TODO: display cell["execution_count"]
    print("")
    if show_source:
        print("```python")
        print("".join(cell["source"]))
        print("```")
    for output in cell["outputs"]:
        if output["output_type"] == "execute_result" and show_execute_results:
            if "text/html" in output["data"]:
                print("".join(output["data"]["text/html"]))
            else:
                print("```")
                print("".join(output["data"]["text/plain"]))
                print("```")
        elif output["output_type"] == "display_data":
            print("".join(output["data"]["text/plain"]))
            print(
                '<img src="data:image/png;base64,{}" />'.format(
                    output["data"]["image/png"].strip()
                )
            )
        else:
            # TODO: raise?
            pass
    print("")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="output a jupyter notebook as remark.js slides",
        prog="python -m zgulde.nbslides",
    )
    parser.add_argument(
        "--show-source",
        action="store_true",
        default=False,
        help="Include the python source code in the slides. default: %(default)s",
    )
    parser.add_argument(
        "--show-execute-results",
        action="store_true",
        default=False,
        help="Include plaintext output from python code in the slides. default: %(default)s",
    )
    parser.add_argument(
        "--html-boilerplate",
        action="store_true",
        default=False,
        help="Include html boilerplate in output. default: $(default)s",
    )
    parser.add_argument("notebook", help="path to the jupyter notebook file")

    args = parser.parse_args()

    if not os.path.exists(args.notebook):
        print("ERROR: could not find file: %s" % args.notebook)
        sys.exit(1)

    notebook = json.load(open(args.notebook))

    if args.html_boilerplate:
        print(
            '<html><head><meta charset="utf-8"/><title>My Presentation</title></head><body><textarea id="source">'
        )

    # we assume the first cell is the title slide
    title_slide = notebook["cells"][0]
    display_markdown(
        title_slide,
        output_slide_seperator=False,
        classes=["center", "middle", "text-invert"],
    )

    for cell in notebook["cells"][1:]:
        if cell["cell_type"] == "markdown":
            display_markdown(cell)
        elif cell["cell_type"] == "code":
            display_code(
                cell,
                show_source=args.show_source,
                show_execute_results=args.show_execute_results,
            )
        else:
            # TODO: handle raw data type
            pass

    if args.html_boilerplate:
        print(
            """
            </textarea>
            <script src="https://remarkjs.com/downloads/remark-latest.min.js"></script>
            <script>var slideshow = remark.create({slideNumberFormat: ''})</script>
            </body>
            </html>
            """
        )
