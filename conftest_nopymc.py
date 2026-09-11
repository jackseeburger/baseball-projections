"""Make pymc/arviz/pytensor look genuinely absent, the way
`requirements-ci.txt` leaves them.

Two halves, because the suite depends on both behaviours:

* `importlib.util.find_spec("pymc")` must return **None**, since that is what
  every `needs_pymc` marker in this suite tests. Raising instead turns a skip
  into a collection error, which is not what CI sees.
* an actual `import pymc` must fail, so that anything importing it outside a
  guard is caught rather than silently working.

Not part of the suite; loaded explicitly with `-p conftest_nopymc` to check
the CI contract locally.
"""
import importlib.util
import sys

BLOCKED = {"pymc", "arviz", "pytensor"}
_real_find_spec = importlib.util.find_spec


def _find_spec(name, package=None):
    if name.split(".")[0] in BLOCKED:
        return None
    return _real_find_spec(name, package)


importlib.util.find_spec = _find_spec


class _Blocker:
    """Refuses the real import, so an unguarded one is loud."""

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None


sys.meta_path.insert(0, _Blocker())
for _mod in list(sys.modules):
    if _mod.split(".")[0] in BLOCKED:
        del sys.modules[_mod]
