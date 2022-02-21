# Full path to database (will be created if it does not exist yet)
DATABASE_NAME = '/media/lynxens/HemoCellData/Projects/HemoCell/hemocell.db'

# Full path to directory to save matrices in (will be created if it does not exist yet)
MATRIX_DIRECTORY = '/tmp/hemocell_matrices'

# Full path to the directory to save the output of the experiments in
EXPERIMENT_OUTPUT_DIRECTORY = '/tmp/hemocell_experiments'

# The fields you want to retrieve from the config.xml file and their corresponding name in the Config Entity.
# (Always collects dx and dt)
CONFIG_FIELDS = [
    ('warmup', 'warmup'),
    ('stepMaterialEvery', 'stepMaterialEvery'),
    ('stepParticleEvery', 'stepParticleEvery'),
    ('Re', 'Re'),
    ('kRep', 'kRep'),
    ('RepCutoff', 'RepCutoff'),
    ('tmax', 'tmax')
]

# The columns you want to extract from the CSV files and their corresponding name in the CSVCell Entity
CSV_CELL_FIELDS = [
    ('X', 'position_x'),
    ('Y', 'position_y'),
    ('Z', 'position_z'),
    ('area', 'area'),
    ('volume', 'volume'),
    ('atomic_block', 'atomic_block'),
    ('cellId', 'cell_id'),
    ('velocity_x', 'velocity_x'),
    ('velocity_y', 'velocity_y'),
    ('velocity_z', 'velocity_z')
]

# The fields you want to retrieve from the fluid HDF5 files and their corresponding name in the HDF5Fluid Entity
HDF5_FLUID_FIELDS = [
    ('Density', '_density'),
    ('Force', '_force'),
    ('ShearRate', '_shear_rate'),
    ('ShearStress', '_shear_stress'),
    ('Velocity', '_velocity')
]

# The fields you want to retrieve from the cell HDF5 files and their corresponding name in the HDF5Cell Entity
HDF5_CELL_FIELDS = [
    ('Area force', 'area_force'),
    ('Bending force', 'bending_force'),
    ('Inner link force', 'inner_link_force'),
    ('Link force', 'link_force'),
    ('Position', 'position'),
    ('Repulsion force', 'repulsion_force'),
    ('Total force', 'total_force'),
    ('Velocity', 'velocity'),
    ('Viscous force', 'viscous_force'),
    ('Volume force', 'volume_force')
]