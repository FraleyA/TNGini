# TNGini
This repository contains scripts for analyzing morphological features of known galaxy mergers in IllustrisTNG.

The package dependencies include a modified version of Illustris and arepo_python_tools folder includes the necessary modifications to the original Illustris code alongside a modified version of
the arepo python tools code.

The main modification to the original scripts includes the addition of the parameter 'subbox_num' as my work revolves around using the
subbox subhalo list catalogue. 'subbox_num' just allows for the specification of which subbox files you want to access in your base_path
directory.

A plot showing clear evolutionary stages in the G-M20 correlation over the course of a merger:
![TNGini_defaultXY](https://github.com/user-attachments/assets/f84802cf-7145-4c36-a71a-a6814d381c69)

The same merger rotated by polar and azimuthal angles (theta, phi) relative to the default viewing plane:
![TNGini_rotatedXY](https://github.com/user-attachments/assets/17e5e4fc-ab0e-4348-ae59-243565cf47d0)
