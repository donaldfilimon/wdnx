import json

import click

from wdbx.client import WDBX
from wdbx.exceptions import WDBXError  # Assuming WDBXError exists


# Helper function for consistent error handling and output
def handle_wdbx_call(ctx, func, success_message, *args, **kwargs):
    """
    Calls a WDBX client function, handles errors, and prints output based on --json flag.
    """
    try:
        result = func(*args, **kwargs)
        if ctx.obj.get("JSON_OUTPUT"):
            click.echo(json.dumps({"status": "success", "result": result}, indent=2))  # noqa: E501
        else:
            click.echo(f"SUCCESS: {success_message}")
            if result is not None and not isinstance(result, bool):  # Don't print None or simple bools unless JSON
                if isinstance(result, (list, dict)):
                    click.echo(json.dumps(result, indent=2))
                else:
                    click.echo(result)
        return result
    except WDBXError as e:  # Catch specific WDBX errors
        error_info = {
            "status": "error",
            "message": str(e),
            "details": getattr(e, "details", None),
        }
        if ctx.obj.get("JSON_OUTPUT"):
            click.echo(json.dumps(error_info, indent=2))
        else:
            click.echo(f"ERROR: {str(e)}", err=True)
            if hasattr(e, "details") and e.details:
                click.echo(f"Details: {e.details}", err=True)
        ctx.exit(1)
    except Exception as e:  # Catch any other unexpected errors
        error_info = {
            "status": "error",
            "message": "An unexpected error occurred.",
            "details": str(e),
        }
        if ctx.obj.get("JSON_OUTPUT"):
            click.echo(json.dumps(error_info, indent=2))
        else:
            click.echo(f"UNEXPECTED ERROR: {str(e)}", err=True)
        if ctx.obj.get("VERBOSE"):
            import traceback

            click.echo(traceback.format_exc(), err=True)
        ctx.exit(1)


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--host",
    envvar="WDBX_HOST",
    help="WDBX server host. Can also be set via WDBX_HOST env var.",
)
@click.option(
    "--port",
    envvar="WDBX_PORT",
    type=int,
    help="WDBX server port. Can also be set via WDBX_PORT env var.",
)
@click.option(
    "--token",
    envvar="WDBX_TOKEN",
    help="JWT token for authentication. Can also be set via WDBX_TOKEN env var.",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")
@click.option("--json-output", "-j", is_flag=True, help="Output results in JSON format.")
@click.pass_context
def wdbx_cli(ctx, host, port, token, verbose, json_output):
    """WDBX Command Line Interface for interacting with the WDBX server."""
    ctx.ensure_object(dict)
    ctx.obj["HOST"] = host
    ctx.obj["PORT"] = port
    ctx.obj["TOKEN"] = token
    ctx.obj["VERBOSE"] = verbose
    ctx.obj["JSON_OUTPUT"] = json_output

    # Initialize WDBX client here if needed, or per command
    # For now, let each command initialize its own client with host/port from ctx.obj
    # This allows commands to function even if global host/port are not set,
    # if the WDBX client has its own defaults or discovery mechanisms.


@wdbx_cli.command("store")
@click.argument("vector", nargs=-1, type=float, required=True)
@click.option(
    "--metadata",
    "metadata_str",
    default="{}",
    help='JSON string for metadata. Example: \'{"key": "value"}\'',
)
@click.pass_context
def store(ctx, vector, metadata_str):
    """
    Store a vector and its associated metadata.

    Example:

        flask wdbx store 0.1 0.2 0.3 --metadata '{"name": "item1"}'
    """
    try:
        metadata = json.loads(metadata_str)
    except json.JSONDecodeError:
        click.echo("Error: Invalid JSON format for metadata.", err=True)
        ctx.exit(1)

    client = WDBX(host=ctx.obj.get("HOST"), port=ctx.obj.get("PORT"), token=ctx.obj.get("TOKEN"))
    handle_wdbx_call(ctx, client.store, "Vector stored.", list(vector), metadata)


@wdbx_cli.command("search")
@click.argument("vector", nargs=-1, type=float, required=True)
@click.option(
    "--limit",
    default=10,
    type=int,
    show_default=True,
    help="Maximum number of results to return.",
)
@click.pass_context
def search(ctx, vector, limit):
    """
    Search for vectors similar to the query vector.

    Example:

        flask wdbx search 0.1 0.2 0.3 --limit 5
    """
    client = WDBX(host=ctx.obj.get("HOST"), port=ctx.obj.get("PORT"), token=ctx.obj.get("TOKEN"))
    handle_wdbx_call(ctx, client.search, "Search completed.", list(vector), limit=limit)


@wdbx_cli.command("bulk-store")
@click.argument("items_json_or_file")
@click.pass_context
def bulk_store(ctx, items_json_or_file):
    """
    Store multiple vectors in bulk.

    ITEMS_JSON_OR_FILE can be a direct JSON string or a path to a file
    containing a JSON list of {vector: [...], metadata: {...}} objects.

    Example (JSON string):

        flask wdbx bulk-store '[{"vector": [0.1,0.2], "metadata": {"id":1}}, {"vector": [0.3,0.4], "metadata": {"id":2}}]'

    Example (File path):

        flask wdbx bulk-store ./data.json
    """
    try:
        if items_json_or_file.startswith("[") or items_json_or_file.startswith("{"):
            items = json.loads(items_json_or_file)
        else:
            with open(items_json_or_file, "r") as f:
                items = json.load(f)
    except json.JSONDecodeError:
        click.echo(
            f"Error: Invalid JSON format in input or file '{items_json_or_file}'.",
            err=True,
        )
        ctx.exit(1)
    except FileNotFoundError:
        click.echo(f"Error: File not found '{items_json_or_file}'.", err=True)
        ctx.exit(1)

    pairs = [(it.get("vector", []), it.get("metadata", {})) for it in items]
    client = WDBX(host=ctx.obj.get("HOST"), port=ctx.obj.get("PORT"), token=ctx.obj.get("TOKEN"))
    handle_wdbx_call(ctx, client.bulk_store, f"{len(pairs)} vectors stored.", pairs)


@wdbx_cli.command("bulk-search")
@click.argument("vectors_json_or_file")
@click.option(
    "--limit",
    default=10,
    type=int,
    show_default=True,
    help="Max results per query vector.",
)
@click.pass_context
def bulk_search(ctx, vectors_json_or_file, limit):
    """
    Search multiple vectors in bulk.

    VECTORS_JSON_OR_FILE can be a direct JSON string or a path to a file
    containing a JSON list of query vectors.

    Example (JSON string):

        flask wdbx bulk-search '[[0.1,0.2], [0.3,0.4]]' --limit 3

    Example (File path):

        flask wdbx bulk-search ./queries.json
    """
    try:
        if vectors_json_or_file.startswith("["):
            vectors = json.loads(vectors_json_or_file)
        else:
            with open(vectors_json_or_file, "r") as f:
                vectors = json.load(f)
    except json.JSONDecodeError:
        click.echo(
            f"Error: Invalid JSON format in input or file '{vectors_json_or_file}'.",
            err=True,
        )
        ctx.exit(1)
    except FileNotFoundError:
        click.echo(f"Error: File not found '{vectors_json_or_file}'.", err=True)
        ctx.exit(1)

    client = WDBX(host=ctx.obj.get("HOST"), port=ctx.obj.get("PORT"), token=ctx.obj.get("TOKEN"))
    handle_wdbx_call(ctx, client.bulk_search, "Bulk search completed.", vectors, limit=limit)


@wdbx_cli.command("delete")
@click.argument("vector_id", type=int)
@click.pass_context
def delete(ctx, vector_id):
    """
    Delete a stored vector by its ID.

    Example:

        flask wdbx delete 123
    """
    client = WDBX(host=ctx.obj.get("HOST"), port=ctx.obj.get("PORT"), token=ctx.obj.get("TOKEN"))
    handle_wdbx_call(ctx, client.delete, f"Vector ID {vector_id} processed for deletion.", vector_id)


@wdbx_cli.command("update-metadata")
@click.argument("vector_id", type=int)
@click.option(
    "--metadata",
    "metadata_str",
    required=True,
    help='JSON string for new metadata. Example: \'{"key": "new_value"}\'',
)
@click.pass_context
def update_metadata(ctx, vector_id, metadata_str):
    """
    Update the metadata for a stored vector.

    Example:

        flask wdbx update-metadata 123 --metadata '{"status": "processed"}'
    """
    try:
        metadata = json.loads(metadata_str)
    except json.JSONDecodeError:
        click.echo("Error: Invalid JSON format for metadata.", err=True)
        ctx.exit(1)

    client = WDBX(host=ctx.obj.get("HOST"), port=ctx.obj.get("PORT"), token=ctx.obj.get("TOKEN"))
    handle_wdbx_call(
        ctx,
        client.update_metadata,
        f"Metadata updated for vector ID {vector_id}.",
        vector_id,
        metadata,
    )


@wdbx_cli.command("get-metadata")
@click.argument("vector_id", type=int)
@click.pass_context
def get_metadata(ctx, vector_id):
    """
    Retrieve the metadata for a stored vector.

    Example:

        flask wdbx get-metadata 123
    """
    client = WDBX(host=ctx.obj.get("HOST"), port=ctx.obj.get("PORT"), token=ctx.obj.get("TOKEN"))
    handle_wdbx_call(
        ctx,
        client.get_metadata,
        f"Metadata retrieved for vector ID {vector_id}.",
        vector_id,
    )


@wdbx_cli.command("count")
@click.pass_context
def count(ctx):
    """
    Return the total number of vectors stored in the database.

    Example:

        flask wdbx count
    """
    client = WDBX(host=ctx.obj.get("HOST"), port=ctx.obj.get("PORT"), token=ctx.obj.get("TOKEN"))
    handle_wdbx_call(ctx, client.count, "Total vector count retrieved.")


@wdbx_cli.command("ping")
@click.pass_context
def ping(ctx):
    """
    Ping the WDBX server to check its health and availability.

    Example:

        flask wdbx ping
    """
    client = WDBX(host=ctx.obj.get("HOST"), port=ctx.obj.get("PORT"), token=ctx.obj.get("TOKEN"))
    # Ping might not return a complex object, adjust message or use direct call
    try:
        alive = client.ping()
        message = "Server is alive." if alive else "Server did not respond as expected."
        if ctx.obj.get("JSON_OUTPUT"):
            click.echo(
                json.dumps(
                    {
                        "status": "success",
                        "result": {"alive": alive, "message": message},
                    }
                )
            )
        else:
            click.echo(message)
    except Exception as e:
        error_info = {"status": "error", "message": "Ping failed.", "details": str(e)}
        if ctx.obj.get("JSON_OUTPUT"):
            click.echo(json.dumps(error_info, indent=2))
        else:
            click.echo(f"ERROR: Ping failed. {str(e)}", err=True)
        if ctx.obj.get("VERBOSE"):
            import traceback

            click.echo(traceback.format_exc(), err=True)
        ctx.exit(1)


@wdbx_cli.command("flush")
@click.option(
    "--confirm",
    is_flag=True,
    prompt="Are you sure you want to flush all vectors? This cannot be undone.",
    help="Confirm the flush operation.",
)
@click.pass_context
def flush(ctx, confirm):
    """
    Remove ALL vectors from the WDBX database. Requires confirmation.

    Example:

        flask wdbx flush --confirm
    """
    if not confirm:  # Should be caught by prompt, but as a safeguard
        click.echo("Flush operation cancelled. Use --confirm to proceed.", err=True)
        ctx.exit(1)

    client = WDBX(host=ctx.obj.get("HOST"), port=ctx.obj.get("PORT"), token=ctx.obj.get("TOKEN"))
    handle_wdbx_call(ctx, client.flush, "All vectors flushed from the database.")


@wdbx_cli.command("list-shards")
@click.pass_context
def list_shards(ctx):
    """
    List configured shard nodes if sharding is enabled and supported.

    Example:

        flask wdbx list-shards
    """
    client = WDBX(host=ctx.obj.get("HOST"), port=ctx.obj.get("PORT"), token=ctx.obj.get("TOKEN"))
    # This requires the client to have a 'shards' attribute or a method to list them
    try:
        shards = list(client.shards.keys()) if hasattr(client, "shards") and client.shards else []
        if ctx.obj.get("JSON_OUTPUT"):
            click.echo(json.dumps({"status": "success", "result": shards}, indent=2))  # noqa: E501
        else:
            if shards:
                click.echo("Configured shards:")
                click.echo(json.dumps(shards, indent=2))
            else:
                click.echo("No shards configured or sharding not enabled/supported by client.")
    except Exception as e:
        error_info = {
            "status": "error",
            "message": "Failed to list shards.",
            "details": str(e),
        }
        if ctx.obj.get("JSON_OUTPUT"):
            click.echo(json.dumps(error_info, indent=2))
        else:
            click.echo(f"ERROR: Could not retrieve shard list. {str(e)}", err=True)
        if ctx.obj.get("VERBOSE"):
            import traceback

            click.echo(traceback.format_exc(), err=True)
        ctx.exit(1)


@wdbx_cli.command("shards-health")
@click.pass_context
def shards_health(ctx):
    """
    Get the health status of each configured shard node.

    Example:

        flask wdbx shards-health
    """
    client = WDBX(host=ctx.obj.get("HOST"), port=ctx.obj.get("PORT"), token=ctx.obj.get("TOKEN"))
    # Assumes client.check_shards_health() exists and returns a dict
    if not hasattr(client, "check_shards_health"):
        message = "Shard health check not supported by this client."
        if ctx.obj.get("JSON_OUTPUT"):
            click.echo(json.dumps({"status": "error", "message": message}))
        else:
            click.echo(message, err=True)
        ctx.exit(1)

    handle_wdbx_call(ctx, client.check_shards_health, "Shard health status retrieved.")


# Plugin management commands
@wdbx_cli.group("plugin", help="Manage WDBX plugins")
@click.pass_context
def plugin(ctx):
    """Plugin management commands."""
    ctx.ensure_object(dict)


@plugin.command("list")
@click.option("--enabled-only", is_flag=True, help="Show only enabled plugins.")
@click.pass_context
def plugin_list(ctx, enabled_only):
    """List registered plugins."""
    client = WDBX(host=ctx.obj.get("HOST"), port=ctx.obj.get("PORT"), token=ctx.obj.get("TOKEN"))
    handle_wdbx_call(ctx, client.list_plugins, "Plugin list retrieved.", active_only=enabled_only)


@plugin.command("enable")
@click.argument("name")
@click.pass_context
def plugin_enable(ctx, name):
    """Enable a plugin by name."""
    client = WDBX(host=ctx.obj.get("HOST"), port=ctx.obj.get("PORT"), token=ctx.obj.get("TOKEN"))
    handle_wdbx_call(ctx, client.enable_plugin, f"Plugin {name} enabled.", name)


@plugin.command("disable")
@click.argument("name")
@click.pass_context
def plugin_disable(ctx, name):
    """Disable a plugin by name."""
    client = WDBX(host=ctx.obj.get("HOST"), port=ctx.obj.get("PORT"), token=ctx.obj.get("TOKEN"))
    handle_wdbx_call(ctx, client.disable_plugin, f"Plugin {name} disabled.", name)


@plugin.command("reload")
@click.argument("name")
@click.pass_context
def plugin_reload(ctx, name):
    """Reload a plugin by name."""
    client = WDBX(host=ctx.obj.get("HOST"), port=ctx.obj.get("PORT"), token=ctx.obj.get("TOKEN"))
    handle_wdbx_call(ctx, client.reload_plugin, f"Plugin {name} reloaded.", name)


# If this script is run directly for testing (outside Flask context)
if __name__ == "__main__":
    # Need to remove flask-specific @with_appcontext if testing standalone.
    # For Flask CLI plugins, they are typically not run this way.
    # This block is more for a standalone Click app.
    # To make it runnable, one would mock or remove @with_appcontext
    # and ensure WDBX can be instantiated.
    # For now, this is just a placeholder.
    # Example:
    # $ python plugins/wdbx_cli/cli.py --host localhost --port 8000 ping
    wdbx_cli(obj={})
