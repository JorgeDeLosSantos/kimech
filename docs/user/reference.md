# API reference

## Core model

```{automodule} kimech
:members: Mechanism, Link, Ground, Point, RevoluteJoint, PrismaticJoint
```

## Driver and solve

```{autoclass} kimech.KinematicDriver
:members:
```

```{autofunction} kimech.solve
```

## Results

```{autoclass} kimech.KinematicSolution
:members:
```

```{autoclass} kimech.Configuration
:members:
```

## Diagnostics and failures

```{autoclass} kimech.SolveDiagnostics
:members:
```

```{autoclass} kimech.SolveDiagnosticSummary
:members:
```

```{autoclass} kimech.SolveFailureContext
:members:
```

```{autoexception} kimech.KinematicSolveError
```

## Sensitivity

```{autofunction} kimech.driver_sensitivity
```

```{autoclass} kimech.DriverSensitivity
:members:
```

## Topology

```{autoclass} kimech.MechanismTopology
:members:
```
