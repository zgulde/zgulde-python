#!/usr/bin/env python
"""
TODO:

- better layout
- customize colors
- handle a regex subject that looks like the formatting html (replace("<", "\\<")?)
- handle exceptions
- cli for specifying text or file, custom colors
- cleanup code
- multiline? regex flags?
- output groupdict (or numeric equivalent) in another text window?
- flag for all matches vs first match?
"""

from functools import partial
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, VSplit, Window, WindowAlign
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit import HTML
from zgulde.hlmatches import _hl_all_matches, _hl_matches

original_text = """
Mary had a little lamb, little lamb, little lamb.
123 Broadway St. San Antonio, TX 78205
"""

# FIXME
hilightfn = partial(
    _hl_matches,
    start="<u>",
    end="</u>",
    groupstart="<firebrick>",
    groupend="</firebrick>",
)

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
        output = _hl_all_matches(
            input_field.text,
            original_text,
            start="<u>",
            end="</u>",
            groupstart="<firebrick>",
            groupend="</firebrick>",
        )
        # _hl_matches(line) for line in original_text.split('\n')
        # output = highlightfn(input_field.text, original_text)
        output_field.text = HTML(output)
    except:
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
