#!/usr/bin/env python
"""
TODO:

- better layout
- customize colors
- handle a regex subject that looks like the formatting html (replace("<", "\\<")?)
- cli for specifying custom colors
- cleanup code
- multiline? regex flags?
- output groupdict (or numeric equivalent) in another text window?
"""

import re
import sys
from time import strftime

from functools import partial
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, VSplit, Window, WindowAlign
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit import HTML
from zgulde.hlmatches import _hl_all_matches, _hl_matches

DEMO_TEXT = """
Mary had a little lamb, little lamb, little lamb.
123 Broadway St. San Antonio, TX 78205
"""

def log(msg):
    with open('debug.log', 'a') as f:
        f.write('\n[{}] {}'.format(strftime('%Y-%m-%d %H:%M:%S'), msg))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="python -m zgulde.hlmatches")
    input_source_group = parser.add_mutually_exclusive_group(required=True)
    input_source_group.add_argument("-t", "--text")
    input_source_group.add_argument("-f", "--file", type=argparse.FileType("r"))
    input_source_group.add_argument("--demo-text", action='store_true')
    parser.add_argument('-a', '--all', help='Highlight all matches', action='store_true')

    args = parser.parse_args()

    if args.file:
        original_text = args.file.read()
    elif args.text:
        original_text = args.text
    else:
        original_text = DEMO_TEXT

def highlight(regexp, subject):
    fn = partial(
        _hl_all_matches if args.all else _hl_matches,
        regexp,
        start="<u>",
        end="</u>",
        groupstart="<firebrick>",
        groupend="</firebrick>",
    )
    if args.all:
        output = fn(subject)
    else:
        output = '\n'.join([fn(line) for line in subject.split('\n')])

    # TODO: fix this
    try:
        return HTML(output)
    except:
        log('Error when highlighting')
        log('regexp:')
        log(regexp)
        log('subject:')
        log(subject)
        log('output:')
        log(output)
        return subject

input_field = Buffer()
output_field = FormattedTextControl(original_text)

input_window = Window(BufferControl(buffer=input_field), height=2)
output_window = Window(output_field)

body = HSplit(
    [output_window, Window(height=1, char="-", style="class:line"), input_window]
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
        output = highlight(input_field.text, original_text)
        output_field.text = output
    except re.error as e:
        output_field.text = original_text


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
