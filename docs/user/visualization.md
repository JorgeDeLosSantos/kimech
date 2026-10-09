# Visual conventions: links, ground and joints

Kimech renders **kinematic schematics**, not detailed part geometry.
`plot(configuration)` draws one solved configuration and
`animate(solution)` reuses the same visual grammar for an ordered history.

## Ground and mobile links

- **Mobile links** are drawn from their structural points (a line between
  two points or a hub-and-spokes scaffold for three or more).
- **Ground** is an inertial reference body, **not** a bar connecting all
  its attachment points. Its anchors are never joined by a ground scaffold.
- A **revolute joint connected to ground** has a small light-gray triangular
  support directly below its pivot. Other revolute joints keep the circular
  marker but do not receive a ground support.
- A **prismatic joint** keeps its visible guide and rectangular slider,
  including when its guide belongs to ground. No additional ground line is
  drawn between that guide and unrelated supports.
- Extra ground reference points remain separately visible without introducing
  geometry connecting them to the mechanism.

The support triangles are deliberately subdued to emphasize the mechanism's
moving structure. They are geometrically fixed throughout animations; they
do **not** represent added physical bodies or constraints.

## Plot and animate

```python
from kimech.visualization import plot, animate

fig, ax = plot(solution[0])
animation = animate(solution, trace_points=[point_p])
```

A solved `KinematicSolution` is necessary for `animate()`; its frame
interval and `fps` are **presentation settings**, not physical integration.
Trace points are optional. See [Examples](examples.md) for runnable mechanisms,
including the four-bar, slider-crank and Multi-DOF 2R.

## Structural topology is a separate view

```python
from kimech.visualization import plot_topology

fig, ax = plot_topology(mechanism)
```

This drawing shows the body–joint connectivity as a **graph**, not physical
placement. Its ground node is therefore a real graph vertex and should not
be confused with the small fixed-pivot supports in `plot()`.
