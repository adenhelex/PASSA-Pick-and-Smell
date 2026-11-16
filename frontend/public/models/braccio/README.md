# Braccio arm model

## Source CAD
`assets/braccio-cad/` at the repo root holds the official Arduino TinkerKit Braccio
Inventor/DWG pack you provided (`ASTA-BRACCIO-PROVA`, `BASE-INFERIORE`,
`COPERCHIO-BASE-NEW`, `DITO2`, …).

Browsers cannot load `.ipt` / `.dwg` directly. To swap in exact mesh exports later,
open those parts in Autodesk Inventor (or Fusion) and export each link as `.stl` or `.glb`
into `stl/` here.

## Runtime meshes
`stl/` is the TinkerKit Braccio mesh set used by the dashboard (same product as the
Arduino CAD zip), packaged for the web from the community URDF
[grassjelly/ros_braccio_urdf](https://github.com/grassjelly/ros_braccio_urdf).
`braccio_arm.urdf` documents the joint offsets used by `ArmView`.
