# Changelog

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
