"""
python -m zgulde.em --help

TODO:

- preserve indentation after doc marker comments (e.g. multiline list items)
"""
import fileinput
import itertools as it
import sys
import textwrap

FILE_EXTENSIONS = dict(
    sh=dict(comment="#", language="bash"),
    py=dict(comment="#", language="python"),
    js=dict(comment="//", language="javascript"),
    sql=dict(comment="--", language="sql"),
    php=dict(comment="//", language="php"),
)


def prev_and_next(xs):
    prevs, items, nexts = it.tee(xs, 3)
    return zip(
        it.chain([None], prevs), items, it.chain(it.islice(nexts, 1, None), [None])
    )


def make_categorizor(doc_marker):
    def is_code(line):
        return line is not None and not line.startswith(doc_marker) and line != ""

    def is_docs(line):
        return line is not None and line.startswith(doc_marker)

    def categorize(x):
        prev, line, next = x
        if is_docs(line):
            return "docs"
        elif is_code(line):
            return "code"
        elif line == "" and is_code(prev) and is_code(next):
            return "code"
        elif line == "" and is_docs(prev) and is_docs(next):
            return "docs"
        else:
            return "ignore"

    return categorize


def main(fp, language, doc_marker):
    lines = map(lambda line: line.rstrip(), fileinput.input(fp))
    keyfn = make_categorizor(doc_marker)
    chars_to_skip = len(doc_marker)

    for section_type, contents in it.groupby(prev_and_next(lines), keyfn):
        if section_type == "ignore":
            pass
        elif section_type == "code":
            print(f"```{language}")
            for _, line, _ in contents:
                print(line)
            print("```")
            print()
        elif section_type == "docs":
            for _, line, _ in contents:
                print(line[chars_to_skip:].lstrip())
            print()
        else:
            raise ValueError(f"Unknown section type: {section_type}")


def get_defaults(args):
    fp = args.file if args.file is not None else "-"
    extension = fp.split(".")[-1]
    language = args.language
    doc_marker = args.doc_marker

    if language is None and extension in FILE_EXTENSIONS.keys():
        language = FILE_EXTENSIONS[extension]["language"]
    elif language is None:
        language = ""

    if doc_marker is None and extension in FILE_EXTENSIONS.keys():
        doc_marker = FILE_EXTENSIONS[extension]["comment"] + ":"
    elif doc_marker is None:
        doc_marker = "#:"

    return fp, language, doc_marker


def show_defaults():
    print("ext marker language")
    for extension, d in FILE_EXTENSIONS.items():
        print("{:>3} {:^6} {}".format(extension, d["comment"] + ":", d["language"]))


if __name__ == "__main__":
    import argparse
    import textwrap

    description = """
    Transform source code -> markdown.

    Specially marked lines (those that start with the doc_marker) are treated as
    markdown, and everything else is turned into a fenced code block.
    """

    parser = argparse.ArgumentParser()
    parser.prog = "python -m zgulde.em"
    parser.description = textwrap.dedent(description)
    parser.add_argument("file", metavar="FILE", help="filepath", nargs="?")
    parser.add_argument("-m", "--doc-marker", help="indicator for a documentation line")
    parser.add_argument("-l", "--language", help="language for fenced code blocks")
    parser.add_argument(
        "--show-defaults",
        action="store_true",
        help="show defaults for each file extension and exit",
    )
    args = parser.parse_args()

    if args.show_defaults:
        show_defaults()
        sys.exit(0)

    fp, language, doc_marker = get_defaults(args)
    main(fp, language, doc_marker)
