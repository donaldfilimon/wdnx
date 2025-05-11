import json
import subprocess
import sys
from pathlib import Path

import click
from flask import Blueprint
from flask.cli import with_appcontext

from app import wdbx
from plugins.pdf_converter.pdf_to_html import pdf_to_html

# CLI plugin blueprint for terminal integration and management
bp = Blueprint("terminal_cli", __name__, cli_group="term")


@bp.cli.command("convert-pdf")
@click.argument("input_pdf", type=click.Path(exists=True))
@click.argument("output_html", type=click.Path())
@click.option(
    "--workers",
    type=int,
    default=None,
    help="Number of worker threads for parallel processing.",
)
@click.option("--skip-vision", is_flag=True, help="Skip the vision LLM stage.")
@click.option("--extract-tables", is_flag=True, help="Extract tables from PDF pages.")
@click.option("--encrypt-html", is_flag=True, help="Encrypt the output HTML file using AES-GCM.")
@click.option(
    "--anchor-hash",
    "anchor_hash_flag",
    is_flag=True,
    help="Anchor the encrypted HTML hash to the blockchain.",
)
@with_appcontext
def convert_pdf_cli(
    input_pdf,
    output_html,
    workers,
    skip_vision,
    extract_tables,
    encrypt_html,
    anchor_hash_flag,
):
    """
    Convert a PDF file to HTML from the terminal.
    """
    pdf_to_html(
        pdf_path=Path(input_pdf),
        html_path=Path(output_html),
        max_workers=workers,
        skip_vision=skip_vision,
        extract_tables=extract_tables,
        encrypt_html=encrypt_html,
        anchor_hash_flag=anchor_hash_flag,
    )
    click.echo(f"Converted {input_pdf} to {output_html}")


@bp.cli.command("list-anchors")
@with_appcontext
def list_anchors_cli():
    """
    List anchored hashes stored on the blockchain.
    """
    anchors = wdbx.list_anchors()
    for anchor in anchors:
        click.echo(anchor)


@bp.cli.command("check-updates")
@click.option("--record", is_flag=True, help="Record outdated packages into WDBX.")
@with_appcontext
def check_updates(record):
    """
    Check for outdated pip packages and optionally record them in the WDBX database.
    """
    from wdbx.update_utils import get_outdated_packages

    outdated = get_outdated_packages()
    click.echo(json.dumps(outdated, indent=2))
    if record:
        # Use the WDBX client initialized in the Flask app
        try:
            from app import wdbx as wdbx_client

            count = wdbx_client.record_outdated_packages()
            click.echo(f"Recorded {count} outdated packages into WDBX")
        except Exception as e:
            click.echo(f"Error recording updates: {e}", err=True)


@bp.cli.command("generate-docs")
@click.option("--serve", is_flag=True, help="Run local documentation server instead of building.")
@click.option("--clean", is_flag=True, help="Clean site directory before building docs.")
@with_appcontext
def generate_docs(serve, clean):
    """
    Build or serve project documentation using MkDocs.
    """
    args = [sys.executable, "-m", "mkdocs"]
    if serve:
        args.append("serve")
    else:
        args.append("build")
        if clean:
            args.append("--clean")
    try:
        subprocess.run(args, check=True)
        click.echo("Docs generation successful.")
    except subprocess.CalledProcessError as e:
        click.echo(f"Docs generation failed: {e}", err=True)


@bp.cli.command("self-update")
@click.option("--wdbx", "wdbx_flag", is_flag=True, help="Schedule WDBX self-update")
@click.option("--lylex", "lylex_flag", is_flag=True, help="Schedule Lylex self-update")
@click.option("--interval", type=int, required=True, help="Interval in minutes")
@click.option(
    "--repo-dir",
    "repo_dir",
    type=click.Path(),
    default=".",
    help="Repository directory",
)
@click.option("--module-paths", "module_paths", multiple=True, help="Module paths for self-update")
@with_appcontext
def self_update(wdbx_flag, lylex_flag, interval, repo_dir, module_paths):
    """
    Schedule self-update tasks for WDBX and/or Lylex.
    """
    if wdbx_flag:
        from app import wdbx

        wdbx.schedule_self_update(
            interval=interval,
            repo_dir=repo_dir,
            module_paths=list(module_paths) or None,
        )
        click.echo(f"Scheduled WDBX self-update every {interval} minutes for {repo_dir}")
    if lylex_flag:
        from app import lylex_db

        lylex_db.schedule_self_update(
            interval=interval,
            repo_dir=repo_dir,
            module_paths=list(module_paths) or None,
        )
        click.echo(f"Scheduled Lylex self-update every {interval} minutes for {repo_dir}")
    if not wdbx_flag and not lylex_flag:
        click.echo("No component specified for self-update. Use --wdbx or --lylex.", err=True)


@bp.cli.command("git-update")
@click.option("--wdbx", "wdbx_flag", is_flag=True, help="Perform immediate WDBX git update")
@click.option("--lylex", "lylex_flag", is_flag=True, help="Perform immediate Lylex git update")
@click.option(
    "--local-dir",
    "local_dir",
    type=click.Path(),
    default=".",
    help="Local repository directory",
)
@click.option("--module-paths", "module_paths", multiple=True, help="Module paths for git update")
@with_appcontext
def git_update_cli(wdbx_flag, lylex_flag, local_dir, module_paths):
    """
    Trigger immediate git update for WDBX and/or Lylex.
    """
    if wdbx_flag:
        from app import wdbx

        wdbx.git_update(local_dir, module_paths=list(module_paths) or None)
        click.echo(f"Performed WDBX git update in {local_dir}")
    if lylex_flag:
        from app import lylex_db

        lylex_db.git_update(local_dir, module_paths=list(module_paths) or None)
        click.echo(f"Performed Lylex git update in {local_dir}")
    if not wdbx_flag and not lylex_flag:
        click.echo("No component specified for git update. Use --wdbx or --lylex.", err=True)
