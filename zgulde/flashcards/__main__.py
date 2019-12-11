from argparse import ArgumentParser
from os import path, system
from sys import exit
from typing import List, NamedTuple

from IPython import embed


class FlashCard(NamedTuple):
    front: str
    back: str


Deck = List[FlashCard]


def die(msg, code=1):
    print(msg)
    exit(1)


def make_cards(content: str) -> Deck:
    card_contents = content.split("\n===\n")
    faces = [card.strip().split("\n---\n") for card in card_contents]
    cards = [FlashCard(front.strip(), back.strip()) for front, back in faces]
    return cards


def review_cards(cards: Deck):
    for card in cards:
        system("clear")
        print(card.front)
        input("Answer? ")
        print("---")
        print()
        print(card.back)
        input("(c)orrect (i)ncorrect? ")


if __name__ == "__main__":
    parser = ArgumentParser(prog="zgulde.flashcards")
    parser.add_argument("flashcard_file")
    args = parser.parse_args()

    print(args.flashcard_file)

    if not path.exists(args.flashcard_file):
        die(f"Error: could not find file {args.flashcard_file}")

    with open(args.flashcard_file) as f:
        content = f.read()

    cards = make_cards(content)
    review_cards(cards)
