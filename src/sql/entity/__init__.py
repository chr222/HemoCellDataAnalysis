from dataclasses import dataclass, field
import numpy
from inspect import isclass

from src.linalg import Vector3Int, Vector3

from typing import get_type_hints, Dict, List, Annotated, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from src.sql.connection import Connection

# Primary key
primary = {'type': 'primary'}


# Exclude from schema
exclude = {'type': 'exclude'}


# Must be unique
unique = {'type': 'unique'}


# Reference to the parent table
def parent(table: str, column: str):
    return {'type': 'parent', 'table': table, 'column': column}


# Accepted column types
COLUMN_TYPE: Dict[type, str] = {
    int: 'integer',
    float: 'real',
    numpy.ndarray: 'array',
    str: 'text'
}


def parse_variable_type(name: str, value_type: type, additional_args: str = '') -> list:
    try:
        return [f'{name} {COLUMN_TYPE[value_type]}' + additional_args]
    except KeyError:
        if issubclass(value_type, Vector3Int):
            return [f'{name}_{suffix} integer' + additional_args for suffix in 'xyz']

        if issubclass(value_type, Vector3):
            return [f'{name}_{suffix} real' + additional_args for suffix in 'xyz']

        raise RuntimeError(f'Unexpected value type for {name}: {value_type}')


def parse_variable_types(variable_types: Dict[str, Union[type, Annotated]]) -> (list, list):
    """
    Parses variable types into columns
    :param variable_types: Variables in the parent class
    :return: (List of columns, List of relations)
    """

    columns = []
    relations = []

    for name, value_type in variable_types.items():
        try:
            # Handle with metadata
            metadata = value_type.__metadata__[0]

            # Skip excluded variables
            if metadata['type'] == 'exclude':
                continue

            # Add relation
            if metadata['type'] == 'parent':
                relations.append(
                    f"FOREIGN KEY ({name}) REFERENCES {metadata['table']} ({metadata['column']}) ON DELETE CASCADE")

            # Get additional arguments for in the table
            additional_args = ''
            if metadata['type'] == 'primary':
                additional_args = ' PRIMARY KEY'

            if metadata['type'] == 'unique':
                additional_args = ' NOT NULL UNIQUE'

            columns += parse_variable_type(name, value_type.__origin__, additional_args)
            continue
        except AttributeError:
            columns += parse_variable_type(name, value_type)

    return columns, relations


def get_columns_and_values(entity: "Entity", variable_types: Dict[str, Union[type, Annotated]]):
    """
    Gets the columns and values from the entity by checking which variables need to be written to the database
    :param entity: The Entity that needs to be inserted
    :param variable_types: The variable types of the entity
    :return: (List of the columns, List of the values)
    """

    columns = []
    values = []

    for name, value_type in variable_types.items():
        if hasattr(value_type, '__metadata__'):
            # Skip excluded variables
            if value_type.__metadata__[0]['type'] == 'exclude':
                continue

            # __origin__ contains the variable type inside the Annotated wrapper
            value_type = value_type.__origin__

        # Split the Vector3 and Vector3Int classes into three separate columns
        if issubclass(value_type, Vector3Int) or issubclass(value_type, Vector3):
            value: Union[Vector3Int, Vector3] = getattr(entity, name)
            columns += [f'{name}_{suffix}' for suffix in 'xyz']
            values += [value.x, value.y, value.z]
            continue

        columns.append(name)
        values.append(getattr(entity, name))

    return columns, values


def get_columns(variable_types: Dict[str, Union[type, Annotated]]):
    """
    Get the columns in the entity
    :param variable_types: Variables in the entity with their types
    :return: The columns that are in the table
    """

    columns = []

    for name, value_type in variable_types.items():
        if hasattr(value_type, '__metadata__'):
            # Skip excluded variables
            if value_type.__metadata__[0]['type'] == 'exclude':
                continue

            # __origin__ contains the variable type inside the Annotated wrapper
            value_type = value_type.__origin__

        if issubclass(value_type, Vector3Int) or issubclass(value_type, Vector3):
            columns += [f'{name}_{suffix}' for suffix in 'xyz']
            continue

        columns.append(name)

    return columns


def get_parent_column(variable_types: Dict[str, Union[type, Annotated]]):
    """
    Get the column that contains the foreign key to the parent table
    """

    for name, value_type in variable_types.items():
        if hasattr(value_type, '__metadata__') and value_type.__metadata__[0]['type'] == 'parent':
            return name

    raise RuntimeError("Could not find a column with a link to the parent table.")


class EntityNotFoundException(Exception):
    def __init__(self, classname, parent_column, parent_id):
        super().__init__(f"Could not find {classname} by {parent_column}={parent_id}")


@dataclass
class Entity:
    # Initialize later, otherwise other class cannot be created on top of it
    id: Annotated[int, primary] = field(default=None, init=False)

    @classmethod
    def from_dict(cls, **kwargs):
        keys = set(kwargs.keys())

        cls_params = {}

        for f, v_type in get_type_hints(cls).items():
            if f == 'id':
                continue  # id has to be set after initialization

            if f in keys:
                cls_params[f] = kwargs[f]
                continue

            if hasattr(v_type, '__metadata__'):
                # __origin__ contains the variable type inside the Annotated wrapper
                v_type = v_type.__origin__

            if isclass(v_type) and issubclass(v_type, Vector3Int):
                # Combine the columns into a Vector3Int
                param_set = [f + '_' + suffix for suffix in 'xyz']
                if set(param_set).issubset(keys):
                    cls_params[f] = Vector3Int(*[kwargs[param] for param in param_set])
                continue

            if isclass(v_type) and issubclass(v_type, Vector3):
                # Combine the columns into a Vector3
                param_set = [f + '_' + suffix for suffix in 'xyz']
                if set(param_set).issubset(keys):
                    cls_params[f] = Vector3(*[kwargs[param] for param in param_set])
                continue

        entity = cls(**cls_params)

        # Set id after initialization
        if 'id' in kwargs:
            entity.id = kwargs['id']

        return entity

    @classmethod
    def __table__(cls):
        """
        Get the table name in snake case
        :return:
        """

        name = cls.__name__

        table_name = ''

        c: str
        for i, c in enumerate(name):
            if 0 < i < len(name) - 1:
                next_c: str = name[i + 1]
                if (c.isnumeric() or c.isupper()) and (not next_c.isnumeric() and not next_c.isupper()):
                    table_name += '_'

            table_name += c.lower()

        return table_name

    @classmethod
    def get_schema(cls) -> str:
        """
        Get the schema of the entity
        """

        columns, relations = parse_variable_types(get_type_hints(cls, include_extras=True))

        return f'CREATE TABLE {cls.__table__()} ({", ".join(columns)}{", " + ", ".join(relations) if relations else ""});'

    def insert(self, connection: "Connection"):
        """
        Insert the entity as a row into the table
        """

        columns, values = get_columns_and_values(self, get_type_hints(self.__class__, include_extras=True))

        self.id = connection.insert(
            f"INSERT INTO {self.__table__()} ({', '.join(columns)}) VALUES ({', '.join(['?'] * len(columns))});",
            tuple(values)
        )

    @classmethod
    def insert_many(cls, connection: "Connection", entities: List["Entity"]):
        """
        Insert multiple entities in one query
        """

        variable_types = get_type_hints(cls, include_extras=True)

        columns = get_columns(variable_types)
        values = [tuple(get_columns_and_values(entity, variable_types)[1]) for entity in entities]

        connection.insert_many(
            f"INSERT INTO {cls.__table__()} ({', '.join(columns)}) VALUES ({', '.join(['?'] * len(columns))});",
            values
        )

    @classmethod
    def load(cls, connection: "Connection", parent_id: int):
        """
        Load an entity by its parent id
        """

        variable_types = get_type_hints(cls, include_extras=True)
        columns = [*get_columns(variable_types)]
        parent_column = get_parent_column(variable_types)

        params = connection.select_one(
            f"SELECT {', '.join(columns)} FROM {cls.__table__()} WHERE {parent_column}=?",
            parent_id
        )

        if params is None:
            raise EntityNotFoundException(cls.__name__, parent_column, parent_id)

        return cls.from_dict(**dict(zip(columns, params)), connection=connection)

    @classmethod
    def load_all(cls, connection: "Connection", parent_id: int):
        """
        Load all entities linked to the parent_id
        """

        variable_types = get_type_hints(cls, include_extras=True)
        columns = [*get_columns(variable_types)]
        parent_column = get_parent_column(variable_types)

        param_sets = connection.select_all(
            f"SELECT {', '.join(columns)} FROM {cls.__table__()} WHERE {parent_column}=?",
            parent_id
        )

        return [cls.from_dict(**dict(zip(columns, params)), connection=connection) for params in param_sets]

    @classmethod
    def get_property_type(cls, property_name: str) -> type:
        variable_types = get_type_hints(cls)
        value_type = variable_types[property_name]

        if hasattr(value_type, '__metadata__'):
            return value_type.__origin__

        return value_type
