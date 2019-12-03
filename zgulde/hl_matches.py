from IPython.display import display, HTML as nb_HTML
from prompt_toolkit import print_formatted_text, HTML as cli_HTML
import re


def hl_all_matches(
    regexp: str,
    subject: str,
    start: str,
    end: str,
    groupstart: str,
    groupend: str,
    initial_output="",
    closing_output="",
) -> str:
    """
    Does not handle nested groups.
    """

    output = ""

    m = re.search(regexp, subject)
    while m is not None:
        mstart, mend = m.span()
        output += hl_matches(
            regexp,
            subject[:mend],
            start,
            end,
            groupstart,
            groupend,
            initial_output="",
            closing_output="",
        )
        subject = subject[mend:]
        if subject == "":
            break
        m = re.search(regexp, subject)

    return initial_output + output + subject + closing_output


def hl_matches(
    regexp: str,
    subject: str,
    start: str,
    end: str,
    groupstart: str,
    groupend: str,
    initial_output="",
    closing_output="",
) -> str:
    """
    Does not handle nested groups.
    """

    m = re.search(regexp, subject)

    if m is None:
        print("No matches!")
        return subject

    n_matches = len(m.groups())
    overall_start, overall_end = m.span()

    spans = [m.span(n) for n in range(1, n_matches + 1)]
    starts = [span[0] for span in spans]
    ends = [span[1] for span in spans]

    output = initial_output

    for i, c in enumerate(subject):
        if i == overall_start:
            output += start

        if i in starts:
            output += groupstart

        if i in ends:
            output += groupend

        if i == overall_end:
            output += end

        output += c

    if len(subject) in ends:
        output += groupend
    if len(subject) == overall_end:
        output += end

    output += closing_output
    return output


def hl_matches_cli(regexp, subject):
    """
    Highlight a regular expressions matches in a string.

    Currently only works in terminal environments (read: this function won't
    work in a jupyter notebook) and does not handle nested groups.

    Uses python's `re.search` under the hood.

    The entire match will be highlighted in bold, the capture groups will be
    red.
    """
    output = hl_matches(
        regexp,
        subject,
        start="<b>",
        end="</b>",
        groupstart="<firebrick>",
        groupend="</firebrick>",
    )
    print_formatted_text(cli_HTML(output))


def hl_matches_nb(regexp, subject):
    """
    Highlight a regular expressions matches in a string. Does not handle nested
    groups. Uses python's `re.search` under the hood.

    The entire match will be underlined, and the capture groups will be red.
    """
    output = hl_matches(
        regexp,
        subject,
        start='<span style="text-decoration: underline">',
        end="</span>",
        groupstart='<span style="color: firebrick">',
        groupend="</span>",
        initial_output='<div style="font-family: monospace; letter-spacing: 3px; font-size: 24px; line-height: 36px;">',
        closing_output="</div>",
    )
    return nb_HTML(output)

def hl_matches_plaintext(regexp, subject):
    '''
    Highlightes a regular expression's matches in `subject` with plaintext
    markers.

    >>> from zgulde.hl_matches import hl_matches_plaintext as hl
    >>> hl(r'.+', 'abc')
    '[abc]'
    >>> hl(r'.', 'abc')
    '[a]bc'
    >>> hl(r'.(.).', 'abc')
    '[a(b)c]'
    >>> hl(r'^(\d+).*?(\d+)$', '123 broadway st san antonio tx 78205')
    '[(123) broadway st san antonio tx (78205)]'
    '''
    return hl_matches(
        regexp,
        subject,
        start="[",
        end="]",
        groupstart="(",
        groupend=")",
        initial_output="",
        closing_output="",
    )


def hl_all_matches_plaintext(regexp, subject):
    '''
    Highlightes all of a regular expression's matches in `subject` with
    plaintext markers.

    >>> from zgulde.hl_matches import hl_all_matches_plaintext as hl
    >>> hl(r'.+', 'abc')
    '[abc]'
    >>> hl(r'.', 'abc')
    '[a][b][c]'
    >>> hl(r'.(.).', 'abc')
    '[a(b)c]'
    >>> hl(r'^(\d+).*?(\d+)$', '123 broadway st san antonio tx 78205')
    '[(123) broadway st san antonio tx (78205)]'
    '''
    return hl_all_matches(
        regexp,
        subject,
        start="[",
        end="]",
        groupstart="(",
        groupend=")",
        initial_output="",
        closing_output="",
    )


def hl_all_matches_nb(regexp, subject):
    """
    Highlight a regular expressions matches in a string. Does not handle nested
    groups.

    Uses python's `re.search` under the hood.

    The entire match will be underlined, and the capture groups will be red.
    """
    output = hl_all_matches(
        regexp,
        subject,
        start='<span style="text-decoration: underline">',
        end="</span>",
        groupstart='<span style="color: firebrick">',
        groupend="</span>",
        initial_output='<div style="font-family: monospace; letter-spacing: 3px; font-size: 24px; line-height: 36px;">',
        closing_output="</div>",
    )
    return nb_HTML(output)
