Custom model example
====================

1. Copy ``docs/model_template.py`` next to your code (e.g., ``my_models.py``) and edit the two required callables:

   .. code-block:: python

      # my_models.py
      import jax.numpy as jnp
      from mrax.data_struct.model import Model

      def make_my_model():
          def model_apply(fields, model_axis, observables=None):
              # model_axis/observables can be None for purely spatial models
              x = model_axis[:, None, None]
              return fields["amp"][None, ...] * jnp.exp(-x / fields["tau"][None, ...])

          def parametrization_factory(spatial_shape):
              params = {"amp_raw": jnp.zeros(spatial_shape), "tau_raw": jnp.zeros(spatial_shape)}

              def unpack(p):
                  amp = jnp.exp(p["amp_raw"])
                  tau = 1e-3 + 5.0 * jnp.exp(p["tau_raw"])
                  return {"amp": amp, "tau": tau}

              return params, unpack

          return Model(
              name="my_decay",
              model_apply=model_apply,
              parametrization_factory=parametrization_factory,
              parameter_units={"amp": "a.u.", "tau": "s"},
          )

2. Use it in code or notebooks:

   .. code-block:: python

     from my_models import make_my_model
     from mrax.core.loss_builder import run_fit, make_model_callable

     model = make_my_model()
     fields, losses, model_fn = run_fit(
         experiment,
         model,
         dataset,
         num_steps=200,
         step_size=0.02,
         return_model_callable=True,
     )
     preds = model_fn(field_arrays)  # merges study globals + observables

3. Hook it into a Study or Bruker import:

   .. code-block:: python

     custom_map = {"my_experiment_type": make_my_model}
     study, meta = import_bruker_study(root, model_map_by_type=custom_map)

The same pattern applies to synthetic studies: construct your ``Model`` first, then
pass it into ``Study`` entries or ``run_fit`` directly. For imported data, use
``model_map_by_type`` or ``recipe=`` when the automatically selected model is not the
one you want.
