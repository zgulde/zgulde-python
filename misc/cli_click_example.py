from click import echo
import click


@click.group()
def sayhello():
    pass


@sayhello.command()
def sayhello_world():
    echo("hello, world!")


@sayhello.command()
@click.option("--name", required=True)
def sayhello_name(name):
    echo("Hello, {}!".format(name))


if __name__ == "__main__":
    sayhello()
