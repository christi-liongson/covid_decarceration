import sqlite3
from sqlite3 import Error
import pandas as pd

from create_db import create_connection, create_table, clean_column_names
from populate_db import extract_data, insert_records

import config

import os.path
from os import path

def build_db(TABLES):
    '''
    Calls functions from create_db and populate_db to create ncdoc db, only if the file doesnt already exist
    '''
    # Create connection to database
    conn = create_connection(config.database_name)

    if conn:
    # Test loop
        for table_name in TABLES:
            print("Working on table...", table_name)

            # Make sure all the CSV files are in the data folder!
            # Default dtype will be string - need to convert later
            print("\tReading in table data as pandas DataFrame")
            table_data = pd.read_csv(config.data_folder + table_name + ".csv",
                                        dtype=str)

            columns = table_data.columns
            columns = ', '.join(columns)
            columns = clean_column_names(columns)

            print("\tCreating table in database...")
            create_table(conn, table_name, columns)
            records = extract_data(table_data)
            
            print("\tInserting records into table...")
            insert_records(conn, table_name, columns, records)

    # Close connection
    conn.commit()
    conn.close()



def main():
    if not path.exists(config.database_name):
        build_db(config.TABLES)


if __name__ == '__main__':
    main()
