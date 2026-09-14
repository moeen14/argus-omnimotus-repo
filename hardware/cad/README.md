# Mechanical design files

This directory contains the editable SolidWorks model and a reference drawing for the Argus Omnimotus mechanical assembly.

## Included files

| File | Description |
|---|---|
| [`native/Argus_Omnimotus_Master_Multibody.SLDPRT`](native/Argus_Omnimotus_Master_Multibody.SLDPRT) | Editable SolidWorks master multibody part containing the mechanical design. |
| [`drawings/Argus_Omnimotus_Assembly_Drawing.jpg`](drawings/Argus_Omnimotus_Assembly_Drawing.jpg) | Reference drawing exported from the assembly. |

Open the `.SLDPRT` file in SolidWorks. Because it is a multibody master part, downstream features or references may depend on the SolidWorks version and configuration used to create it. The drawing image is provided for quick viewing without CAD software.

The robot chassis is primarily composed of 3D-printed PLA parts reinforced with two 5 mm steel rods. Individual printable bodies have not yet been released as STL files, and their quantities, orientations, layer settings, infill, and support requirements remain to be documented before fabrication.

The native SolidWorks file is stored with [Git Large File Storage](https://git-lfs.com/) because it exceeds GitHub's standard per-file size limit. Install Git LFS and run `git lfs install` before cloning or committing changes to native CAD files.

Future neutral exchange files should be placed under `step/`, and printable parts under `stl/`. Document units, materials, print orientation, fasteners, and assembly order when adding exported parts.

[Hardware home](../README.md) | [Project resources](../docs/project-links.md)
