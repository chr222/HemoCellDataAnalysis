# HemoCellDataAnalysis
This is a data analysis tool that can be used to parse the output of the [HemoCell framework](https://www.hemocell.eu/).
This is done in two steps. First, the data is imported into a SQLite database. Next, the data can be queried from this 
database using functions in this tool.

## 1. Requirements
- Python 3.9 or higher
- The libraries in [requirements.txt](requirements.txt)

## 2. Setup
To set up this tool the following steps need to be done manually.

1. Download the libraries in [requirements.txt](requirements.txt)
2. Copy [params.dist.py](params.dist.py) and rename it to params.py
3. Set the required variables in params.py
   1. DATABASE_NAME: This is the path to the database that you want to use. If it does not exist yet, it will be created.
   2. CSV_CELL_FIELDS: This is a map of the columns you want to extract from the CSV files and their corresponding name 
you want in the [CSVCell](src/sql/entity/csv_cell.py) Entity.
   3. HDF5_FLUID_FIELDS: This is a map of the fields you want to retrieve from the fluid HDF5 files and their 
corresponding name you want in the [HDF5Fluid](src/sql/entity/hdf5_fluid.py) Entity.
   4. HDF5_CELL_FIELDS: This is a map of the fields you want to retrieve from the cell HDF5 files and their 
corresponding name you want in the [HDF5Cell](src/sql/entity/hdf5_cell.py) Entity

### 2.1 HemoCell changes
Before all features of the tool can be used, some changes need to be made to HemoCell and the code that runs the 
simulation. A [patch](patch/hdf5_output_fix.patch) file has been created that changes some functions in HemoCell to
make sure that the CellID is a scalar in the output instead of a vector. Next to this, it makes a change to the python 
processing script that allows scalars can to be used in ParaView.

The next thing you need to do is add some extra data to the output of your program. To be able to process the cell
HDF5 output, you will need to add the OUTPUT_POSITION, OUTPUT_TRIANGLES and OUTPUT_CELL_ID flags to it. Next, the
OUTPUT_BOUNDARY flag is needed in the fluid output. Now, your setOuputs function should look something like this:
```
void setupOutputs(HemoCell &hemoCell) {
    vector<int> cellOutputs = {
        OUTPUT_POSITION,
        OUTPUT_TRIANGLES,
        OUTPUT_CELL_ID,
        ...
    };
    hemoCell.setOutputs("RBC", cellOutputs);
    hemoCell.setOutputs("PLT", cellOutputs);
    
    hemoCell.setFluidOutputs({
        OUTPUT_BOUNDARY,
        ...
    });
}
```

The last change to your program is the addition of a function that outputs the config params in a JSON format. Furthermore,
it outputs the sizes and offset of the atomic blocks, which are needed to parse the HDF5 fluid data into numpy arrays.
In this function you can also decide what parameters you want to save in the database. The only values that the tool requires
are nx, ny, nz, dx and dt.

```
void printConfigParams(Config *cfg, VoxelizedDomain3D<T> *voxelizedDomain) {
    hlog << "ConfigParams: {" << std::endl;

    SparseBlockStructure3D const& sparseBlock = voxelizedDomain->getVoxelMatrix().getMultiBlockManagement().getSparseBlockStructure();
    plint numBlocks = sparseBlock.getNumBlocks();
    hlog << "\t" << "\"blocks\": {" << std::endl;
    for (const auto &it : sparseBlock.getBulks()) {
        hlog << "\t\t" << "\"" << it.first << "\": {" << std::endl;
        hlog << "\t\t\t" << "\"size\": [" << it.second.getNx() << ", " << it.second.getNy() << ", " << it.second.getNz() << "]," << std::endl;
        hlog << "\t\t\t" << "\"offset\": [" << it.second.x0 << ", " << it.second.y0 << ", " << it.second.z0 << "]" << std::endl;

        if (it.first == numBlocks - 1) {
            hlog << "\t\t" << "}" << std::endl;
        } else {
            hlog << "\t\t" << "}," << std::endl;
        }
    }
    hlog << "\t" << "}," << std::endl;
    hlog << "\t" << "\"nx\": " << sparseBlock.getBoundingBox().getNx() << "," << std::endl;
    hlog << "\t" << "\"ny\": " << sparseBlock.getBoundingBox().getNy() << "," << std::endl;
    hlog << "\t" << "\"nz\": " << sparseBlock.getBoundingBox().getNz() << "," << std::endl;

    hlog << "\t" << "\"warmup\": " << (*cfg)["parameters"]["warmup"].read<int>() << "," << std::endl;
    hlog << "\t" << "\"stepMaterialEvery\": " << (*cfg)["ibm"]["stepMaterialEvery"].read<int>() << "," << std::endl;
    hlog << "\t" << "\"stepParticleEvery\": " << (*cfg)["ibm"]["stepParticleEvery"].read<int>() << "," << std::endl;
    hlog << "\t" << "\"fluidEnvelope\": " << (*cfg)["domain"]["fluidEnvelope"].read<int>() << "," << std::endl;
    hlog << "\t" << "\"rhoP\": " << (*cfg)["domain"]["rhoP"].read<int>() << "," << std::endl;
    hlog << "\t" << "\"nuP\": " << (*cfg)["domain"]["nuP"].read<double>() << "," << std::endl;
    hlog << "\t" << "\"dx\": " << (*cfg)["domain"]["dx"].read<double>() << "," << std::endl;
    hlog << "\t" << "\"dt\": " << (*cfg)["domain"]["dt"].read<double>() << "," << std::endl;
    hlog << "\t" << "\"refDir\": " << (*cfg)["domain"]["refDir"].read<int>() << "," << std::endl;
    hlog << "\t" << "\"refDirN\": " << (*cfg)["domain"]["refDirN"].read<int>() << "," << std::endl;
    hlog << "\t" << "\"blockSize\": " << (*cfg)["domain"]["blockSize"].read<int>() << "," << std::endl;
    hlog << "\t" << "\"kBT\": " << (*cfg)["domain"]["kBT"].read<double>() << "," << std::endl;
    hlog << "\t" << "\"Re\": " << (*cfg)["domain"]["Re"].read<double>() << "," << std::endl;
    hlog << "\t" << "\"particleEnvelope\": " << (*cfg)["domain"]["particleEnvelope"].read<int>() << "," << std::endl;
    hlog << "\t" << "\"kRep\": " << (*cfg)["domain"]["kRep"].read<double>() << "," << std::endl;
    hlog << "\t" << "\"RepCutoff\": " << (*cfg)["domain"]["RepCutoff"].read<double>() << "," << std::endl;
    hlog << "\t" << "\"tmax\": " << (*cfg)["sim"]["tmax"].read<int>() << "," << std::endl;
    hlog << "\t" << "\"tmeas\": " << (*cfg)["sim"]["tmeas"].read<int>() << "," << std::endl;
    hlog << "\t" << "\"tcsv\": " << (*cfg)["sim"]["tcsv"].read<int>() << "," << std::endl;
    hlog << "\t" << "\"tcheckpoint\": " << (*cfg)["sim"]["tcheckpoint"].read<int>() << std::endl;
    hlog << "}" << std::endl;
}
```

### 2.2 Entities
All table in the database are represented by Entity classes. The [Entity](src/sql/entity/__init__.py) class is used to
create a schema from the variables in the class. This schema is then used to create the table. Next, via the `insert`
function, the entities are inserted as rows. These rows can later be queried to recreate the entities via `load` and
`load_all`.

#### 2.2.1 Annotations
Via annotations special properties can be set on the variables. Current these are:
1. `exclude`, used to exclude a variable from the table. This is useful to include other object in an object.
2. `primary`, used to set a variable as the primary key of the table. This is the `id` variable by default.
3. `unique`, used to make sure the value of this column are unique.
4. `parent(table, column)`, used to mark a variable as the foreign key connecting it with another table.

#### 2.2.2 Data types
Each variable can be one of the following datatypes:
- float
- int
- str
- Vector3 (for float vectors of size 3)
- Vector3Int (for int vectors of size 3)

### 2.3 Config
The [Config](src/sql/entity/config.py) Entity contains the configuration output from the simulation. The properties you
want to retrieve from the simulation log can be defined here.

### 2.4 CSVCell
The [CSVCell](src/sql/entity/csv_cell.py) Entity contains the data from the CSV files. The columns you want to retrieve
from the csv output can be defined here. 

To be able to query them later you will need to add the additional variables as property function in the 
[CSVCells](src/sql/entity/csv_cell.py) class.

### 2.5 HDF5Cell
The [Hdf5Cell](src/sql/entity/hdf5_cell.py) Entity contains the cell data from the HDF5 files. HemoCell divides the 
cells into triangles and the forces on each of the triangles are saved in the output. To parse these the program requires
the cell id belonging to each triangle. Otherwise, it won't be able to know what triangles belong together. The required
data from the HDF5 files can be defined in this function. Hereby, it is important that the size of the vectors of the data
is also defined using numpy.empty(0, VECTOR_SIZE).

To be able to later query the data, they can be added as property functions in the 
[Hdf5Cells](src/sql/entity/hdf5_cell.py) class.

### 2.6 HDF5Fluid
The [Hdf5Fluid](src/sql/entity/hdf5_fluid.py) Entity contains the fluid data from the HDF5 files. Unlike the Hdf5Cell
and CSVCell entities, the HDF5Fluid entity contains both its variables and property classes. To allow these to coexist
the variables are prefixed by a '_'. 

If you want to retrieve additional fields from the data, they can be added to the Entity as variables and then a 
property function can be created to retrieve it.


## 3. Import the data
Now the program and data is correctly configured, the data can be imported into the database. This can be done by
running [output_to_database.py](output_to_database.py) with two arguments: the name you want to give the simulation in
the database and the directory that contains the data. For example:
```
python output_to_database.py template_project /path/to/hemocell/simulation/output
```

This script can also be used to write multiple dataset to the database simultaneously.
Furthermore, it is possible to cancel the import process while it is running and continue it later. To do this you need
to run the command again when you want to continue. If the program sees that a project with this name already exists, 
you will be asked if you want to continue importing the data. If you say, yes it will continue where it ended last time.
If you say no, you will be asked if you want to overwrite the data. In this case the data belonging to this simulation 
will be removed and the import progress will start from the beginning.


## 4. Run experiments
Once the data is safely in the database the experiments can begin. To do this the data can be queried and put into its
corresponding classes. To help process the data some support classes are created to get some components using a single
function. For instance, the Vector3Matrix can be used to calculate the velocity magnitude using the magnitude property.

Classes with similar functions are:
- [Tensor9Matrix](src/sql/entity/hdf5_fluid.py)
- [Tensor6Matrix](src/sql/entity/hdf5_fluid.py)
- [Hdf5CellData](src/sql/entity/hdf5_cell.py)
- [CSVCellData](src/sql/entity/csv_cell.py)

The classes they contain can be found in the [linalg](src/linalg/__init__.py) package.

Finally, the data can be visualized. To help with this the [graphics](src/graphics/__init__.py) package can be used, but
this is not necessary.


## 5. Additional info
Some additional info on the project can be found in this section

### 5.1 Project structure
- patch/; a directory containing patches for HemoCell.
- src/; the python source code directory.
  - experiments/; a directory for the experiments.
  - graphics/; a package that can be used to visualize the data using matplotlib.
  - linalg/; a package that contains some additional data classes to represent that data.
  - progress/; a package with classes used to visualize the progress of functions.
  - sql/; a directory that contains the entities used to represent the data in the database and a script to interact with the SQLite3 database.
- tests/; tests to check whether the tool works as expected. It can be run using the pytests directory.
- output_to_database.py; a script to parse the data and insert it into the database.
- params.dist.py; a template file for the parameters file required to run the script (params.py).
- README.md; an explanation of the tool.
- requirements.txt; a list of the required python packages
- run_experiments.py; an example script to run experiments on the data in the database


### 5.2 Connection class
To interact with the database the framework uses entities in combination with the [Connection](src/sql/connection.py) 
class. The connection class is used to do insertions, queries and deletions in the database.

### 5.3 Bulk Collector
The [BulkCollector](src/sql/entity/bulk_collector.py) class can be used to query a lot of data in batches. It can be
used by calling the `all` property of the [Simulation](src/sql/entity/simulation.py) Entity.


