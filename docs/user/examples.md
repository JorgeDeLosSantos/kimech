# Examples

The repository includes executable examples in the `examples/` directory.

## Four-bar

`examples/four_bar.py` demonstrates position solving and animation for a
classical four-bar linkage.

`examples/four_bar_analysis.py` adds angular velocity, angular acceleration,
and coupler-point differential kinematics.

## Slider-crank

`examples/slider_crank.py` demonstrates a crank-slider mechanism.

`examples/slider_crank_analysis.py` shows differential analysis and also
demonstrates prescribing the prismatic coordinate instead of the crank angle.

`examples/slider_crank_analysis_comparison.py` compares Kimech against an
independent closed-form slider-crank solution.

## More demanding studies

The `playground/` directory contains larger mechanisms and numerical studies,
including Whitworth, Watt six-bar, Klann, Theo Jansen, scale robustness,
singularity diagnostics, and branch-continuity experiments. These are valuable
engineering studies but are intentionally separate from the compact public
user guide.
