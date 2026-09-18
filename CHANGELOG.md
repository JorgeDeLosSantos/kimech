# Changelog

## 0.3.0

Kimech `0.3.0` consolidates numerical robustness and simplifies the public solve API.

### Changed

- `solve()` now uses `input_joint` and `input_position` instead of `input` and `values`;
- `solve()` always returns `KinematicSolution`, including scalar prescribed positions;
- scalar prescribed positions produce a one-sample solution and `solution[0]` returns the corresponding `Configuration`;
- `Configuration.input_value` was renamed to `Configuration.input_position`;
- `KinematicSolution.input_values` was renamed to `KinematicSolution.input_positions`;
- position, velocity, and acceleration solves use internal dimensionless numerical scaling and scale-independent residual validation;
- point queries are named consistently as `point_position(s)`, `point_velocity/velocities`, and `point_acceleration/accelerations`;
- `KinematicSolution` supports slicing to another `KinematicSolution` and explicit iteration over `Configuration` objects;
- result containers remain externally immutable and may be empty when constructed manually;
- prescribed-input metadata on `Configuration` requires an associated `input_joint`;
- visualization uses explicit body render scaffolds with fallback support for plate-like links and lightweight connectors for auxiliary points;
- joint glyph scaling is derived from body scaffolds rather than arbitrary remote auxiliary points;
- `animate()` supports opt-in progressive point traces through `trace_points=[...]`.

These are intentional breaking changes. No compatibility aliases are provided because Kimech has not yet established an external user base or PyPI release workflow.

### Still out of scope

- multiple prescribed inputs or a public Driver abstraction;
- automatic initial-guess generation or branch discovery;
- formal singularity diagnostics;
- dynamics, forces, torques, masses, or inertias.


## 0.2.0

Kimech `0.2.0` extends the initial position-kinematics MVP with differential kinematics for the same one-DOF planar rigid-body mechanisms with revolute and prismatic joints.

### Added

- prescribed input velocity and acceleration through `solve()`;
- generalized coordinate velocity and acceleration state;
- body velocity and acceleration queries;
- point velocity and acceleration queries;
- joint velocity and acceleration queries;
- velocity and acceleration histories in `KinematicSolution`;
- analytic differential kinematics using the existing constraint Jacobian;
- independent residual verification for velocity and acceleration linear solves;
- acceptance coverage for four-bar and slider-crank mechanisms, including prismatically driven inverse kinematics.

### Changed

- `Configuration.pose()` was renamed to `Configuration.body_pose()`;
- `KinematicSolution.link_poses()` was renamed to `KinematicSolution.body_poses()`;
- result objects can now retain optional `q_dot` / `q_ddot` state and prescribed differential input metadata.

The two method renames are intentional breaking changes. No compatibility aliases are provided because Kimech is still pre-`1.0`.

### Still out of scope

- multiple prescribed inputs;
- formal singularity diagnostics;
- adaptive continuation or automatic branch discovery;
- dynamics, forces, torques, masses, or inertias;
- explicit time arrays or motion-law objects;
- additional visualization backends.

## 0.1.0

Initial position-kinematics MVP for one-DOF planar rigid-body mechanisms with revolute and prismatic joints, including warm-start continuation, result queries, plotting, and animation.
