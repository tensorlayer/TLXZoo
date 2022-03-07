import schema
import tensorlayerx as tlx
from schema import Schema
import numpy as np
from dataclasses import dataclass


class TaskDataSetSchema(Schema):
    def __init__(self, schema_type, *args, **kwargs):
        self.schema_type = schema_type
        if isinstance(schema_type, dict) and isinstance(list(schema_type.values())[0], SchemaType):
            _schema = {j.name: j.schema for j in schema_type.values()}
        else:
            _schema = schema_type
        super(TaskDataSetSchema, self).__init__(_schema, *args, **kwargs)

    def get_dtypes(self):
        return [i.dtype for i in self.schema_type.values()]

    def get_names(self):
        return [i.name for i in self.schema_type.values()]


@dataclass
class SchemaType:
    name: str = None
    dtype: type = None
    schema: type = None


image_classification_task_data_set_schema = TaskDataSetSchema(
    {"img": SchemaType(name="img", dtype=tlx.float32, schema=np.ndarray),
     "label": SchemaType(name="label", dtype=tlx.int64, schema=np.int64)}, ignore_extra_keys=True)

object_detection_task_data_set_schema = TaskDataSetSchema(
    {"img": SchemaType(name="img", dtype=tlx.float32, schema=np.ndarray),
     "label": SchemaType(name="label", dtype=tlx.float32, schema=tuple)}, ignore_extra_keys=True)


if __name__ == '__main__':
    d = {"img": np.array([1.0, 2.0]), "label": np.array([1])[0].astype("int64")}

    print(image_classification_task_data_set_schema.validate(d))
