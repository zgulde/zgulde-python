from prompt_toolkit import print_formatted_text, HTML
import re

def hl_matches(regexp, subject):
    '''
    from zgulde.hl_matches import hl_matches

    note the hl_matches function uses python's `regexp.search` under the hood.

    >>> hl_matches(r'^.', 'abc')
    >>> hl_matches(r'^(\d).*?(\d+)', '123 broadway st 78205 san antonio tx')

    The entire match will be highlighted in bold, the capture groups will be
    red.
    '''
    m = re.search(regexp, subject)

    if m is None:
        print('No matches!')
        return

    n_matches = len(m.groups())
    overall_start, overall_end = m.span()

    spans = [m.span(n) for n in range(1, n_matches + 1)]
    starts = [span[0] for span in spans]
    ends = [span[1] for span in spans]

    output = ''

    for i, c in enumerate(subject):
        if i == overall_start:
            output += '<b>'

        if i in starts:
            output += '<firebrick>'

        if i in ends:
            output += '</firebrick>'

        if i == overall_end:
            output += '</b>'

        output += c

    if len(subject) in ends:
        output += '</firebrick>'
    if len(subject) == overall_end:
        output += '</b>'

    print_formatted_text(HTML(output))
