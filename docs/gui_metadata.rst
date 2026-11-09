Metadata
========

Open **View → Metadata**. Select an experiment
in the study browser, then choose experiment metadata, study constants, study
overview, arrays or diagnostics. Study overview also renames the study and experiment.

Click groups and breadcrumb paths to inspect nested data. Search finds field
names, paths, values and units across the categories. Tables preserve value types;
**Raw** edits the complete selected document as JSON and remains available if a
table cannot be built. Summaries, arrays and diagnostics are read-only.

Use **Add field** or **Add item** to extend the selected object or list. Custom
experiment fields belong under **metadata**; quantities used by models belong
under **observables**, with a numeric value or array and a unit.

**Import from file** accepts JSON, CSV and TSV for the selected group. Review
the preview and choose whether to update matching fields or add only new fields,
then select **Import to draft**. JSON contains the group's
value. CSV and TSV use ``name`` (or ``path``) and ``value`` columns, with optional
``type`` and ``unit`` columns. For example:

.. code-block:: text

   name,value,type,unit
   operator,Research team,string,
   repetition_time,1500,number,ms
   reviewed,false,boolean,

``path`` uses JSON pointers relative to the selected group, such as
``/sequence/repetition_time``. Escape a slash within a field name as ``~1`` and
a tilde as ``~0``. Without ``type``, JSON values are inferred and other values
remain text; use ``string`` to preserve identifiers. A unit creates a quantity
with ``values`` and ``unit`` fields. Imports do not modify the source file.

Edits remain in a separate draft for each experiment or study until **Apply**.
Switching tasks or experiments retains drafts. **Revert draft** reloads the
current values. If metadata changes elsewhere, resolve the conflict before
applying. Save the workspace to retain applied changes on disk.
