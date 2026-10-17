Model parameters
================

Default bounds, transforms, units, and initial values are defined in
``mrax/library/parameter_library.toml``. Each ``make_*_model`` factory can select a named
parameter set via ``param_key`` (default ``"default"``) and apply targeted overrides via
``param_overrides``:

.. code-block:: python

   from mrax.library.default_models import make_t1_ir_model

   model = make_t1_ir_model(
       param_key="default",
       param_overrides={
           "t1": {"bounds": (100.0, 4000.0)},
           "amplitude": {"init_raw": 0.5},
       },
   )

Use ``init`` when you want to specify the starting value in constrained physical units;
it takes precedence if an inherited default also defines ``init_raw``. Built-in defaults
use physical ``init`` values throughout. Scale-relative softplus amplitudes use
``init_scale_key`` to keep the physical initialization tied to the model scale. Use
``init_raw`` only when you deliberately want to initialize the unconstrained optimizer
coordinate directly.

The parameter library is plain TOML. You can point to another file with
``MRAX_PARAMETER_LIBRARY``. For local pool definitions outside the public defaults, set
``MRAX_PRIVATE_PARAMETER_LIBRARY`` to an overlay TOML file; its contents are merged on
top of the public library.
