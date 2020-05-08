import sqlite3
from sqlite3 import Error
import re

from create_db import create_connection, clean_column_names

def extract_data(table_data):
    '''
    Extracts tuples from data
    Inputs:
        - table_data: pandas DataFrame

    Returns: list of tuples to use in insert_records function
    '''
    return list(table_data.itertuples(index=False, name=None))


def insert_records(conn, table_name, columns, records):
    '''
    Insert multiple records of data into table
    Inputs:
        - conn: Connection object
        - table_name: name of the table / csv e.g. INMT4AA1
        - columns: list of column names from the csv
        - records: list of tuples to insert into table

    Returns: None (updates table in database)
    '''
    try:
        c = conn.cursor()
        sql = ('INSERT INTO {}({}) VALUES({})').format(table_name, columns)
        c.executemany(sql, list_records)

    except Error as e:
        print(e)
