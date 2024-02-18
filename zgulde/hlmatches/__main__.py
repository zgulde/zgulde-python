"""
TODO:

- better layout / layout options (e.g. regex at top or at bottom?, like fzf)
- customize colors / styles for highlighting
- different colors for different capture groups
- cleanup code
- multiline? regex flags?
- output groupdict (or numeric equivalent) in another text window?
- word wrap? probably as a cli arg
"""

import re
import sys
from functools import partial
from time import strftime

from prompt_toolkit import HTML
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, VSplit, Window, WindowAlign
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.layout import Layout

DEMO_TEXT = """
Mary had a little lamb, little lamb, little lamb.
123 Broadway St. San Antonio, TX 78205
127.0.0.1 - - [16/May/2020 10:48:11] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [16/May/2020 10:48:11] code 404, message File not found
123.123.123.123 - - [12/May/2020:13:37:02 +0000] "GET /api/v1/items HTTP/1.1" 200 3561 "-" "python-requests/2.23.0"
"""


def log(msg):
    with open("debug.log", "a") as f:
        f.write("\n[{}] {}".format(strftime("%Y-%m-%d %H:%M:%S"), msg))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="python -m zgulde.hlmatches")
    input_source_group = parser.add_mutually_exclusive_group(required=True)
    input_source_group.add_argument("-t", "--text")
    input_source_group.add_argument("-f", "--file", type=argparse.FileType("r"))
    input_source_group.add_argument("--demo-text", action="store_true")
    parser.add_argument(
        "-a", "--all", help="!Experimental! Highlight all matches", action="store_true"
    )

    args = parser.parse_args()

    if args.file:
        original_text = args.file.read().strip()
    elif args.text:
        original_text = args.text.strip()
    else:
        original_text = DEMO_TEXT.strip()

# TODO: looks like there's some duplicattion between hlmatch and hlmatches,
# maybe we could extract a common function here.
def hlmatch(regexp, subject, start, end, groupstart, groupend):
    """
    >>> from functools import partial
    >>> hl = partial(hlmatches, start='[', end=']', groupstart='(', groupend=')')
    >>> hl(r'.+', 'abc')
    HTML('[abc]')
    >>> hl(r'.', 'abc')
    HTML('[a]bc')
    >>> hl(r'.(.).', 'abc')
    HTML('[a(b)c]')
    >>> hl(r'^(\d+).*?(\d+)$', '123 broadway st san antonio tx 78205')
    HTML('[(123) broadway st san antonio tx (78205)]')
    """
    match = re.search(regexp, subject)
    if (
        not match
        or len(regexp) == 0
        or len(subject) == 0
        or match.start() == match.end()
    ):
        return "".join(["{}"] * len(subject)), subject
    output = ["{}"] * len(subject)
    n_groups = len(match.groups())
    if n_groups > 0:
        for i in range(1, n_groups + 1):
            if len(match.group(i)) == 0:
                continue
            output[match.start(i)] = groupstart + output[match.start(i)]
            output[match.end(i) - 1] = output[match.end(i) - 1] + groupend
    output[match.start()] = start + output[match.start()]
    output[match.end() - 1] = output[match.end() - 1] + end
    return "".join(output), subject


def hlmatches(regexp, subject, start, end, groupstart, groupend):
    matches = list(re.finditer(regexp, subject, re.MULTILINE))
    if not matches or len(regexp) == 0 or len(subject) == 0:
        return "".join(["{}"] * len(subject)), subject
    output = ["{}"] * len(subject)
    for match in matches:
        if match.start() == match.end():
            continue
        n_groups = len(match.groups())
        if n_groups > 0:
            for i in range(1, n_groups + 1):
                if len(match.group(i)) == 0:
                    continue
                output[match.start(i)] = groupstart + output[match.start(i)]
                output[match.end(i) - 1] = output[match.end(i) - 1] + groupend
        output[match.start()] = start + output[match.start()]
        output[match.end() - 1] = output[match.end() - 1] + end
    return "".join(output), subject


def highlight(regexp, subject):
    fn = partial(
        hlmatches if args.all else hlmatch,
        regexp,
        start="<u>",
        end="</u>",
        groupstart="<firebrick>",
        groupend="</firebrick>",
    )
    if args.all:
        # output = HTML(fn(subject))
        output_fmt, subject = fn(subject)
        output = HTML(output_fmt).format(*subject)
    else:
        highlights = [fn(line) for line in subject.split("\n")]
        output_fmt = "\n".join([h[0] for h in highlights])
        subjects = "".join([h[1] for h in highlights])
        output = HTML(output_fmt).format(*subjects)

    try:
        return output
    except Exception as e:
        # TODO: we probably shouldn't swallow *all* errors here
        return subject


input_field = Buffer()
output_field = FormattedTextControl(original_text)
message_field = FormattedTextControl("Enter regexp below:")

input_window = Window(BufferControl(buffer=input_field), height=1)
message_window = Window(message_field, height=1)
output_window = Window(output_field)

body = HSplit(
    [
        output_window,
        Window(height=1, char="-", style="class:line"),
        message_window,
        Window(height=1, char="-", style="class:line"),
        input_window,
        Window(height=1, char="-", style="class:line"),
    ]
)

title_window = Window(
    height=1,
    content=FormattedTextControl("Press [Ctrl-C] to quit."),
    align=WindowAlign.CENTER,
)

root_container = HSplit(
    [title_window, Window(height=1, char="-", style="class:line"), body]
)


kb = KeyBindings()


@kb.add("c-c", eager=True)
def _(event):
    event.app.exit()


def handle_keypress(_):
    if len(input_field.text) == 0:
        output_field.text = original_text
        return

    try:
        output_field.text = highlight(input_field.text, original_text)
        message_field.text = "Enter regexp below:"
    except re.error as e:
        output_field.text = original_text
        message_field.text = HTML("<red>Error: {}</red>".format(e.msg))


input_field.on_text_changed += handle_keypress

application = Application(
    layout=Layout(root_container, focused_element=input_window),
    key_bindings=kb,
    mouse_support=False,
    full_screen=True,
)


def run():
    application.run()


if __name__ == "__main__":
    run()
