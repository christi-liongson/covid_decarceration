import sqlite3
import pandas as pd
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


def insert_records(conn, table, columns, records):
    '''
    Insert multiple records of data into table
    Inputs:
        - conn: Connection object
        - table (str): name of the table / csv e.g. INMT4AA1
        - columns (str): column names from the csv
        - records (list): tuples from extract_data to insert into table

    Returns: None (updates table in database)
    '''
    try:
        c = conn.cursor()
        mulitplier = len(columns.split(", "))
        sql = 'INSERT INTO {} ({}) VALUES '.format(table, columns)
        values = '%s' + (', %s' * multiplier)
        sql += '({});'.format(values)
        c.executemany(sql, records)

    except Error as e:
        print(e)


# For testing
test_df = pd.DataFrame({'name': ['John', 'Karen'], 'age': [41, 32]})
columns = test_df.columns
columns = ', '.join(columns)

sql_create_test_table = """DROP TABLE IF EXISTS test;
                           CREATE TABLE IF NOT EXISTS test (name,age);"""
