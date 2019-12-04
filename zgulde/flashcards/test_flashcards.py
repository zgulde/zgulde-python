import textwrap as tw

import pytest

from zgulde.flashcards.__main__ import FlashCard, make_cards


def test_card_parsing():
    content = tw.dedent(
        """
    here is the front
    ---
    and this is the back
    ===
    this is another card
    ---
    this is the second back
    """
    ).strip()

    cards = make_cards(content)

    assert len(cards) == 2

    first, second = cards

    assert first.front == "here is the front"
    assert first.back == "and this is the back"

    assert second.front == "this is another card"
    assert second.back == "this is the second back"


def test_whitespace_doesnt_matter_in_card_parsing():
    content = tw.dedent(
        """

    here is the front

    ---

    and this is the back

    ===

    this is another card


    ---

    this is the second back
    """
    ).strip()

    cards = make_cards(content)

    assert len(cards) == 2

    first, second = cards

    assert first.front == "here is the front"
    assert first.back == "and this is the back"

    assert second.front == "this is another card"
    assert second.back == "this is the second back"
