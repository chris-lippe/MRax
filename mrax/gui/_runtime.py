"""Set desktop allocator defaults before GUI imports initialize JAX."""

import os

# Browsing images must not reserve most of the GPU's memory. XLA reads this
# when its backend is first initialized, including by module-level arrays.
# Preserve allocator settings explicitly supplied by the caller.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
