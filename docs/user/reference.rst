API reference
=============

This page documents the public symbols exported by :mod:`kimech`.

Core model
----------

.. autoclass:: kimech.Mechanism
   :members:

.. autoclass:: kimech.Link
   :members:

.. autoclass:: kimech.Ground
   :members:

.. autoclass:: kimech.Point
   :members:

.. autoclass:: kimech.RevoluteJoint
   :members:

.. autoclass:: kimech.PrismaticJoint
   :members:

Driver and solve
----------------

.. autoclass:: kimech.KinematicDriver
   :members:

.. autofunction:: kimech.solve

Results
-------

.. autoclass:: kimech.KinematicSolution
   :members:

.. autoclass:: kimech.Configuration
   :members:

Diagnostics and validation
--------------------------

.. autoclass:: kimech.SolveDiagnostics
   :members:

.. autoclass:: kimech.SolveDiagnosticSummary
   :members:

.. autoclass:: kimech.SolveFailureContext
   :members:

.. autoclass:: kimech.ValidationReport
   :members:

Errors
------

.. autoexception:: kimech.KimechError

.. autoexception:: kimech.InvalidModelError

.. autoexception:: kimech.KinematicSolveError

Sensitivity
-----------

.. autofunction:: kimech.driver_sensitivity

.. autoclass:: kimech.DriverSensitivity
   :members:

Topology
--------

.. autoclass:: kimech.MechanismTopology
   :members:
