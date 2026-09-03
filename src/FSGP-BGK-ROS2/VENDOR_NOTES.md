# FSGP-BGK Vendor Notes

This directory keeps the FSGP-BGK algorithm source used as the reference for
PTR-Nav integration.

Upstream:

```text
https://github.com/MarineRock10/FSGP-BGK-ROS2.git
```

Local cleanup for PTR-Nav:

- The upstream nested `.git` directory was moved out of the workspace backup.
- The upstream ROS1/ROS2 demo simulation packages and bundled PCD demo maps were
  moved out of the workspace backup.
- The algorithm code under `src/fsgp_bgk` is kept as vendor/reference source.
- PTR-Nav runtime integration lives in `src/fsgp_bgk_se2`.

Reason:

PTR-Nav already has its own Isaac/Gazebo Hunter simulation, FAST-LIO2 mapping,
terrain analyzer, and terrain navigation stack. Keeping the upstream demo
simulator in the main workspace would add unrelated packages and large PCD
assets without improving the integration chain.
