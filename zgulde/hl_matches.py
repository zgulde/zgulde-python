from IPython.display import display, HTML as nb_HTML
from prompt_toolkit import print_formatted_text, HTML as cli_HTML
import re
from typing import Iterator, Tuple, List


MatchIndexes = List[int]

def _starts_ends_overall(m: re.Match) -> Tuple[MatchIndexes, MatchIndexes, MatchIndexes]:
    '''
    Extracts indices from a match object.

    Returns

    (groupstarts, groupends, [overall_start, overall_end])

    >>> m = re.match(r'.(.)', 'abc')
    >>> _starts_ends_overall(m)
    ([1], [2], [0, 2])
    >>> m = re.match(r'.', 'abc')
    >>> _starts_ends_overall(m)
    ([], [], [0, 1])
    '''
    overall_start, overall_end = m.span()
    n_matches = len(m.groups())

    spans = [m.span(n) for n in range(1, n_matches + 1)]
    starts = [span[0] for span in spans]
    ends = [span[1] for span in spans]
    return starts, ends, [overall_start, overall_end]


def _starts_ends_overalls(matches: Iterator[re.Match]) -> Tuple[MatchIndexes, MatchIndexes, MatchIndexes, MatchIndexes]:
    '''
    >>> matches = re.finditer(r'.', 'abc')
    >>> _starts_ends_overalls(matches)
    ([], [], [0, 1, 2], [1, 2, 3])
    >>> matches = re.finditer(r'.(.)', 'abc')
    >>> _starts_ends_overalls(matches)
    ([1], [2], [0], [2])
    >>> matches = re.finditer(r'.(.)', 'abcdef')
    >>> _starts_ends_overalls(matches)
    ([1, 3, 5], [2, 4, 6], [0, 2, 4], [2, 4, 6])
    '''
    groupstarts = []
    groupends = []
    overallstarts = []
    overallends = []
    for starts, ends, (start, end) in map(_starts_ends_overall, matches):
        groupstarts += starts
        groupends += ends
        overallstarts += [start]
        overallends += [end]
    return groupstarts, groupends, overallstarts, overallends

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

    matches = re.finditer(regexp, subject)

    groupstarts, groupends, overallstarts, overallends = _starts_ends_overalls(matches)

    output = initial_output

    for i, c in enumerate(subject):
        if i in groupends:
            output += groupend
        if i in overallends:
            output += end
        if i in overallstarts:
            output += start
        if i in groupstarts:
            output += groupstart

        output += c

    if len(subject) in groupends:
        output += groupend
    if len(subject) in overallends:
        output += end

    output += closing_output
    return output

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

    overall_start, overall_end = m.span()
    n_matches = len(m.groups())

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

    The entire match will be underlined, the capture groups will be red.
    """
    output = hl_matches(
        regexp,
        subject,
        start="<u>",
        end="</u>",
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
    """
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
    """
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
    """
    Highlightes all of a regular expression's matches in `subject` with
    plaintext markers.

    >>> from zgulde.hl_matches import hl_all_matches_plaintext as hl
    >>> hl(r'.+', 'abc')
    '[abc]'
    >>> hl(r'.', 'abc')
    '[a][b][c]'
    >>> hl(r'..', 'abc')
    '[ab]c'
    >>> hl(r'.(.).', 'abc')
    '[a(b)c]'
    >>> hl(r'^(\d+).*?(\d+)$', '123 broadway st san antonio tx 78205')
    '[(123) broadway st san antonio tx (78205)]'
    >>> hl(r'^.', 'abc')
    '[a]bc'
    """
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

def hl_all_matches_cli(regexp, subject):
    """
    Highlight a regular expressions matches in a string.

    Currently only works in terminal environments (read: this function won't
    work in a jupyter notebook) and does not have special highlighting for
    nested groups.

    Uses python's `re.finditer` under the hood.

    The entire match will be underlined, the capture groups will be red.
    """
    output = hl_all_matches(
        regexp,
        subject,
        start="<u>",
        end="</u>",
        groupstart="<firebrick>",
        groupend="</firebrick>",
    )
    print_formatted_text(cli_HTML(output))
