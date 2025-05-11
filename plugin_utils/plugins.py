def plugin_metadata(name: str, version: str, author: str, description: str = ''):
    """
    Decorator to attach PLUGIN_INFO metadata to a plugin module or class.

    Args:
        name: Human-readable plugin name.
        version: Plugin version string.
        author: Plugin author name or identifier.
        description: Short description of the plugin.
    """
    def decorator(obj):
        setattr(obj, 'PLUGIN_INFO', {
            'name': name,
            'version': version,
            'author': author,
            'description': description
        })
        return obj
    return decorator 