from flask import Blueprint
from flask.cli import with_appcontext
import click
import json
from lylex.db import LylexDB

bp = Blueprint('lylex_cli', __name__, cli_group='lylex')

@bp.cli.command('store-interaction')
@click.argument('prompt')
@click.argument('response')
@click.option('--metadata', default='{}', help='JSON metadata')
@with_appcontext
def store_interaction(prompt, response, metadata):
    """Store a prompt-response interaction"""
    db = LylexDB()
    vid = db.store_interaction(prompt, response, json.loads(metadata))
    click.echo(vid)

@bp.cli.command('search-interactions')
@click.argument('prompt')
@click.option('--limit', default=5, help='Max results')
@with_appcontext
def search_interactions(prompt, limit):
    """Search similar interactions by prompt"""
    db = LylexDB()
    results = db.search_interactions(prompt, limit)
    click.echo(json.dumps(results, indent=2))

@bp.cli.command('count-interactions')
@with_appcontext
def count_interactions():
    """Return count of stored interactions"""
    db = LylexDB()
    count = db.count_interactions()
    click.echo(count)

@bp.cli.command('ping-backend')
@with_appcontext
def ping_backend():
    """Ping the Lylex backend for health check"""
    db = LylexDB()
    alive = db.ping_backend()
    click.echo(alive)

@bp.cli.command('flush-interactions')
@with_appcontext
def flush_interactions():
    """Remove all stored interactions"""
    db = LylexDB()
    db.flush_interactions()
    click.echo('Flushed all interactions')

@bp.cli.command('export-interactions')
@with_appcontext
def export_interactions():
    """Export all stored interactions as JSON."""
    db = LylexDB()
    data = db.export_interactions()
    click.echo(json.dumps(data, indent=2))

@bp.cli.command('list-shards')
@with_appcontext
def list_shards():
    """List configured shard nodes."""
    db = LylexDB()
    shards = list(db.client.shards.keys()) if hasattr(db.client, 'shards') else []
    click.echo(json.dumps(shards, indent=2))

@bp.cli.command('shards-health')
@with_appcontext
def shards_health():
    """Get health status of each shard."""
    db = LylexDB()
    health = db.client.check_shards_health() if hasattr(db.client, 'check_shards_health') else {}
    click.echo(json.dumps(health, indent=2))

@bp.cli.command('check-updates')
@with_appcontext
def check_updates():
    """
    Check for updates and record them in the WDBX client.
    """
    if record:
        # Use the WDBX client initialized in the Flask app
        try:
            from app import wdbx as wdbx_client
            count = wdbx_client.record_outdated_packages()
            click.echo(f'Recorded {count} outdated packages into WDBX')
        except Exception as e:
            click.echo(f'Error recording updates: {e}', err=True)

@bp.cli.command('brain-status')
@with_appcontext
def brain_status():
    """
    Show whether the Brain autonomous learner is scheduled.
    """
    from app import brain
    job = brain.scheduler.get_job(f'brain_{brain.model_name}')
    click.echo('scheduled' if job else 'not scheduled')

@bp.cli.command('brain-stop')
@with_appcontext
def brain_stop():
    """
    Stop the Brain autonomous learning.
    """
    from app import brain
    brain.stop()
    click.echo('Brain stopped.')

@bp.cli.command('brain-start')
@with_appcontext
def brain_start():
    """
    Start the Brain autonomous learning.
    """
    from app import brain
    try:
        brain.scheduler.start()
        click.echo('Brain started.')
    except Exception as e:
        click.echo(f'Error starting Brain: {e}', err=True)

@bp.cli.command('brain-sweep')
@click.argument('param')
@click.argument('values', nargs=-1, type=float)
@click.option('--param2', default=None, help='Name of the second parameter for 2D sweep.')
@click.option('--values2', multiple=True, type=float, help='Values for the second parameter (for 2D sweep).')
@with_appcontext
def brain_sweep(param, values, param2, values2):
    """Run a parameter sweep in Brain's NEURON simulator."""
    from app import brain
    try:
        # Convert tuples to lists for CLI args
        processed_values = list(values)
        processed_values2 = list(values2) if values2 else None

        if param2 and not processed_values2:
            click.echo("Error: --values2 must be provided if --param2 is specified.", err=True)
            return
        if not param2 and processed_values2:
            click.echo("Error: --param2 must be provided if --values2 is specified.", err=True)
            return

        results = brain.simulate_sweep(param, processed_values, param_name2=param2, values2=processed_values2)

        # Helper to serialize trace data (numpy arrays to lists)
        def serialize_trace_data(trace_tuple):
            t, v = trace_tuple
            return {'time': t.tolist(), 'values': v.tolist()}

        # Serialize traces for output, handling 1D or 2D sweep results
        output = {}
        if param2 and processed_values2: # 2D sweep
            for val1, nested_results in results.items():
                output[str(val1)] = {str(val2): serialize_trace_data(trace_data) for val2, trace_data in nested_results.items()}
        else: # 1D sweep
            output = {str(val): serialize_trace_data(trace_data) for val, trace_data in results.items()}

        click.echo(json.dumps(output, indent=2))
    except Exception as e:
        click.echo(f'Error sweeping Brain: {e}', err=True) 