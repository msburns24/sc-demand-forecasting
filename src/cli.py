"""
Simple factory functions for Typer/Rich
"""

from typing import Optional

from typer import Context, Typer
from rich.console import Console
from rich.panel import Panel
from rich.padding import Padding
from rich.pretty import Pretty
from rich.table import Table
from rich.text import Text


console = Console(force_jupyter=False, force_terminal=True)


def create_app(
    name: Optional[str] = None,
    help: Optional[str] = None,
    **kwargs,
) -> Typer:
    """
    Creates a Typer app with better defaults, namely:

    - Default help options as both `-h` and `--help`, not just `--help`
    - Don't add completion; i.e., `add_completion = False`

    These new defaults can be overwritten via keyword arguments (`kwargs`).
    """
    kwargs["name"] = name
    kwargs["help"] = help

    # Use `dict.setdefault()` to allow overwrite via **kwargs
    kwargs.setdefault("context_settings", {"help_option_names": ["-h", "--help"]})
    kwargs.setdefault("add_completion", False)
    return Typer(**kwargs)


def print_options(ctx: Context) -> None:
    params = ctx.params
    table = Table.grid(padding=(0, 2))
    for key, value in params.items():
        table.add_row(Text(f"{key}:", style="dim italic"), Pretty(value))

    panel = Panel(
        table,
        title=Text("Options", style="dim"),
        title_align="left",
        border_style="dim",
        expand=False,
    )
    console.print(Padding(panel, (1, 1)))
    return
