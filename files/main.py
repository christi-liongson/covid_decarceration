import sqlite3
from sqlite3 import Error
import pandas as pd

from create_db import create_connection, create_table, clean_column_names
from populate_db import extract_data, insert_records

import config

TABLES = ["APPT7AA1", "AAPT9BJ1", "INMT4AA1", "INMT4BB1", "INMT4CA1",
          "INMT9CF1", "OFNT1BA1", "OFNT3AA1", "OFNT3BB1", "OFNT3CE1",
          "OFNT3DE1", "OFNT9BE1"]

def main():
    # Create connection to database
    conn = create_connection(config.database_name)

    if conn is not None:

    # Next, we should create a populate_db file, and call functions here
    # to insert values into inmate table
    # if that works fine, then we can write a function (in main?)
    # which loops through all our tables and creates them like inmate table above

    # Test loop
        for table_name in TABLES:

            # Make sure all the CSV files are in the data folder!
            table_data = pd.read_csv(config.data_folder + table_name + ".csv")

            columns = table_data.columns
            columns = ', '.join(columns)
            columns = clean_column_names(columns)

            create_table(conn, table_name, columns)
            records = extract_data(table_data)
            insert_records(conn, table_name, columns, records)

    # Close connection
    conn.commit()
    conn.close()

    # NOTE: Ideally, we would create table columns and specify things like
    # primary key, data type, etc. Without this, chances are all data
    # will be read in like text. however, with hundreds of columns,
    # that seems pretty difficult. I think eventually we'll
    # want to create an object that groups our variables together anyway
    # in this we can group "Numeric vars" and during preprocessing turn
    # them to a numeric format for actual analysis?

if __name__ == '__main__':
    main()
