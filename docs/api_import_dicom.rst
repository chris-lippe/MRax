DICOM import functions
======================

Discover and load classic or enhanced DICOM image series. The general layer handles
pixel scaling, functional groups, geometry, slice/frame ordering, dimensions, and
standard MR observables. Vendor adapters add origin-specific metadata;
the shared DICOM core owns pixel decoding and geometry.

.. autoclass:: mrax.io.dicom.DicomSeriesInfo
.. autoclass:: mrax.io.dicom.DicomMetadataPatch
.. autoclass:: mrax.io.dicom.DicomMetadataAdapter
.. autoclass:: mrax.io.dicom_bruker.BrukerDicomMetadataAdapter
.. autofunction:: mrax.io.dicom.discover_dicom_series
.. autofunction:: mrax.io.dicom.load_dicom_series
.. autofunction:: mrax.io.dicom.load_dicom_study
