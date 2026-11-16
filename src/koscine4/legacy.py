"""The only door to the older code (requirement N-3).

`koscine4` may import a module from the old `koscine`, `koscine3`, `pipeline` or `analysis`
packages only if it has been reviewed and is listed here. `tests/v1/test_package_rules.py`
enforces this by scanning every import in the package.
"""

LEGACY_PACKAGES = ("koscine", "koscine3", "pipeline", "analysis")

# Fully qualified module names, each added only after review. Empty at phase 0.
REVIEWED_LEGACY_MODULES: tuple[str, ...] = ()
