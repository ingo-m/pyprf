"""Get the version of the installed pyprf package (set in pyproject.toml)."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version('pyprf')
except PackageNotFoundError:
    __version__ = ('Version information not found. Please install this '
                   + 'project through pip.')
