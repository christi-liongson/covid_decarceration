import sqlite3
from sqlite3 import Error
import pandas as pd

from create_db import create_connection, create_table, clean_column_names
from populate_db import extract_data, insert_records

import config

table_name = "OFNT3CE1"

conn = create_connection(config.database_name)


print("\tReading in table data as pandas DataFrame")
table_data = pd.read_csv(config.data_folder + table_name + ".csv",
                            dtype=str)

columns = table_data.columns
columns = ', '.join(columns)
columns = clean_column_names(columns)

print("\tCreating table in database...")
print(columns)
create_table(conn, table_name, columns)
print("\tExtract records")
records = extract_data(table_data)

print("\tInserting records into table...")
insert_records(conn, table_name, columns, records)