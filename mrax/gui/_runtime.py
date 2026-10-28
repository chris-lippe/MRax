"""Set desktop allocator defaults before GUI imports initialize JAX."""

import os

# Use on-demand GPU allocation for image browsing. XLA reads this
# when its backend is first initialized, including by module-level arrays.
# Preserve allocator settings explicitly supplied by the caller.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
