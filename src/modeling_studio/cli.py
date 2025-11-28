"""
CLI entry points.
"""

import typer

app = typer.Typer(help="Modeling Studio CLI")


def train():
    """Train a model."""
    pass


def evaluate():
    """Evaluate a model."""
    pass


if __name__ == "__main__":
    app()
