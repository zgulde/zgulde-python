from .__main__ import make_categorizor, prev_and_next
import textwrap
import os
import pytest
from subprocess import check_output

@pytest.fixture
def extract_markdown():
    def _extract_markdown(input, expected, args=None):
        cmd = 'python -m zgulde.em'
        input = textwrap.dedent(input).strip()
        expected = textwrap.dedent(expected).strip()
        if args is not None:
            cmd += args
        output = check_output(cmd.split(),
                              input=input.encode()).decode().strip()
        print('----- actual')
        print(output)
        print('----- expected')
        print(expected)
        return output, expected
    yield _extract_markdown

def test_categorizor():
    lines = textwrap.dedent('''
    #: # Hello, World
    #: 
    #: This is a demo.

    foo

    bar

    baz
    ''').strip().split('\n')
    categorize = make_categorizor('#:')
    categories = [categorize(line) for line in prev_and_next(lines)]
    assert categories == ['docs'] * 3 + ['ignore'] + ['code'] * 5

def test_docs_and_code(extract_markdown):
    actual, expected = extract_markdown('''
        #: docs
        #: docs
        #: docs
        code
        code
        ''',
        '''
        docs
        docs
        docs

        ```
        code
        code
        ```
        '''
    )
    assert actual == expected

def test_just_docs(extract_markdown):
    actual, expected = extract_markdown('''
        #: docs
        ''',
        '''
        docs
        '''
    )
    assert actual == expected


def test_just_code(extract_markdown):
    actual, expected = extract_markdown('''
        code
        ''',
        '''
        ```
        code
        ```
        '''
    )
    assert actual == expected

def test_blank_lines_are_preserved_within_docs_or_code(extract_markdown):
    actual, expected = extract_markdown('''
        #: docs
        #:
        #: docs

        code

        code
        ''',
        '''
        docs

        docs

        ```
        code

        code
        ```
        '''
    )
    assert actual == expected

def test_blank_lines_between_docs_and_code_are_auto_inserted_if_not_present(extract_markdown):
    actual, expected = extract_markdown('''
        #: docs
        code
        #: docs
        code
        ''',
        '''
        docs

        ```
        code
        ```

        docs

        ```
        code
        ```
        '''
    )
    assert actual == expected

def test_multiple_blank_lines_between_docs_and_code_are_ignored(extract_markdown):
    actual, expected = extract_markdown('''
        #: docs


        code


        #: docs


        code
        ''',
        '''
        docs

        ```
        code
        ```

        docs

        ```
        code
        ```
        '''
    )
    assert actual == expected

def test_2_or_more_blanks_in_code_create_seperate_blocks(extract_markdown):
    actual, expected = extract_markdown('''
        code


        code
        ''',
        '''
        ```
        code
        ```

        ```
        code
        ```
        '''
    )
    assert actual == expected
