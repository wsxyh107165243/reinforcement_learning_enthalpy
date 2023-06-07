
from collections import namedtuple

AtomicPercentage = namedtuple('AtomicPercentage', \
    ('Ti', 'Ni', 'Cu', 'Hf', 'Zr', 'Nb', 'Co', 'Cr', 'Fe', 'Mn', 'Pd'))

AlloyProperty = namedtuple('AlloyProperty', \
    ('enthalpy', 'config_entropy', 'diff_atom_radii', 'ea', 'cs'))

ExperimentPoint = namedtuple('ExperimentPoint', \
    ('AtomicPercentage', 'AlloyProperty'))