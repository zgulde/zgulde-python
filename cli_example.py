"""
An example program demonstrating the usage of the ``argparse`` module and a
setup with the logging module.

    python cli_example.py -h
"""

import logging


def greet(name, greeting, times, exclamation):
    logging.info("performing greeting")
    for n in range(times):
        logging.debug("iteration # %d" % n)
        print("{}, {}{}".format(greeting, name, "!" if exclamation else "."))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="An example program demonstrating python CLIs"
    )
    # Program will default to how the script was invoked, or it can be set
    # explicitly. This is what appears in the help message.
    # Several properties can be set either as a kwarg to the constructor, or like below:
    parser.program = "example_cli"

    # named arguments
    # choices= constrains possible selections
    parser.add_argument(
        "--greeting",
        default="Hello",
        help="Greeting (default: %(default)s)",
        choices=["Hello", "Salutations", "Greetings"],
    )
    # type= can be specified to automatically convert to the correct type
    # (otherwise we get everything as a string)
    parser.add_argument(
        "-n",
        "--n-greetings",
        default=1,
        type=int,
        help="number of greetings (default: %(default)s)",
    )
    # example boolean flag with action='store_true'
    parser.add_argument(
        "--no-happy",
        default=False,
        action="store_true",
        help="do not be happy about the greeting",
    )
    # actions='count' to count the number of occurances of the flag
    parser.add_argument("-v", "--verbose", action="count", help="output verbosity")

    # positional arguments
    parser.add_argument(
        "name",
        metavar="NAME",
        help="name(s) of the person (or people) to greet",
        nargs="+",  # nargs = '+' means 1 or more, also the type will now be a list[str] instead of just str
    )

    # parse the arguments, will grab the cli arguments (i.e. argv) that were
    # passed, you can also pass a list of strings, e.g. for testing
    args = parser.parse_args()

    # args can be accessed through their long name
    n = args.n_greetings
    # args with -s are converted to _s
    happy = not args.no_happy

    # logger setup
    if args.verbose is None:
        loglevel = logging.WARN
    elif args.verbose == 1:
        loglevel = logging.INFO
    elif args.verbose >= 2:
        loglevel = logging.DEBUG
    logging.basicConfig(
        format="%(asctime)s:%(levelname)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=loglevel,
    )

    for name in args.name:
        greet(name, args.greeting, n, happy)
