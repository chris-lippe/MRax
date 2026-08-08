DICOM Import
============

MRax imports classic and enhanced DICOM image objects through ``pydicom``. The importer
is split into two layers:

- :mod:`mrax.io.dicom` implements file discovery, pixel decoding and rescaling,
  functional-group parsing, frame and slice ordering, geometry, standard observables,
  and conversion to MRax containers.
- :mod:`mrax.io.dicom_bruker` adds Bruker origin detection and metadata conventions.
  It does not replace the standards-based image or geometry path.

This separation keeps the general contract vendor-neutral and allows another origin to
be supported by implementing :class:`mrax.io.dicom.DicomMetadataAdapter`.

Loading a series or study
-------------------------

Use :func:`mrax.io.discover_dicom_series` to inspect a source without decoding all pixel
data. A source can be one file, a folder containing multiple files, a folder tree, or a
sequence of paths. Patient-identifying attributes are omitted from discovery summaries.

.. code-block:: python

   from mrax.io import discover_dicom_series, load_dicom_series, load_dicom_study

   series = discover_dicom_series("dicom_folder")
   experiment, dataset, metadata = load_dicom_series("dicom_folder")
   study, study_metadata = load_dicom_study("dicom_study_root")

``load_dicom_series`` requires ``series_instance_uid=...`` when the source contains
more than one series. ``load_dicom_study`` recursively groups all readable image files
by series and produces one short, unique experiment name per group. Set
``recursive=False`` to restrict discovery to the selected directory.

Identifiers are excluded from returned import metadata by default. Set
``include_identifiers=True`` only when the caller has an explicit reason and an
appropriate data-handling context.

Dimensions, observables, and geometry
-------------------------------------

The importer applies DICOM rescale slope/intercept and sorts spatial slices from image
orientation and position metadata. Standard or enhanced functional groups provide echo,
repetition and inversion times, flip angles, diffusion b-values and directions when
present. The shared import-definition layer uses varying observables and frame labels to
infer quantitative experiment types and non-spatial dimensions; otherwise it preserves
a generic ``frame`` axis.

MRax arrays keep fixed ``x/y/z`` spatial semantics. DICOM patient orientation, origin,
and voxel spacing are stored separately as :class:`mrax.data_struct.SpatialGeometry`.
This geometry controls anatomical plane labels, display aspect ratio, and NIfTI affine
conversion, but does not reorder inputs to reconstruction or quantitative fitting.
When valid orientation metadata is unavailable, MRax retains the native array order and
uses explicit fallback geometry rather than guessing anatomy from image dimensions.

Vendor adapters and missing metadata
------------------------------------

The Bruker adapter is selected automatically when the DICOM manufacturer identifies a
Bruker export. It supplements standards-based fields with sequence/frame hints used for
CEST, WASABI, FAIR, and related experiment inference. Only information actually present
in the DICOM objects can be imported. Acquisition fields that remain solely in the
vendor source format are reported as unavailable rather than synthesized.

To supply another adapter explicitly:

.. code-block:: python

   experiment, dataset, metadata = load_dicom_series(
       "dicom_folder",
       metadata_adapters=[MyDicomMetadataAdapter()],
   )

An adapter may recognize a source, select the primary image objects, and return a
:class:`mrax.io.dicom.DicomMetadataPatch`. It should leave general DICOM decoding,
geometry, and standards-based metadata handling to :mod:`mrax.io.dicom`.
