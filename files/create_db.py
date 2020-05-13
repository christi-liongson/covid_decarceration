import sqlite3
from sqlite3 import Error
import pandas as pd
import re

# Code from https://www.sqlitetutorial.net/sqlite-python/create-tables/
# Don't need to specify primary key https://stackoverflow.com/questions/25954543/sqlite-without-primary-key/25954583

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return conn

def clean_column_names(columns):
    '''some of the column names in the NC DOC files have
    characters that are causing errors e.g. "." and "()".
    This function cleans that before creating the table'''
    columns = re.sub(r'[\.\/()&-]','',columns)
    columns = re.sub(r'[#]','NO',columns)
    return columns


def create_table(conn, table_name, columns):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param table_name: name of the table which also corresponds to csv
                       e.g. INMT4AA1
    :param columns: list of column names from the csv
    :return: None. Creates table in database
    """
    try:
        c = conn.cursor()
        c.execute('DROP TABLE IF EXISTS {};'.format(table_name))

        execute = 'CREATE TABLE IF NOT EXISTS {} ({});'.format(table_name,columns)
        c.execute(execute)

    except Error as e:
        print(e)


# def main():
#     database = r"../data/nc_doc.db"

#     # Create inmates table
#     sql_create_inmates_table = """
#                                 DROP TABLE IF EXISTS inmates;
#                                 CREATE TABLE IF NOT EXISTS inmates (
#                                         id,
#                                         name,
#                                         begin_date,
#                                         end_date
#                                     ); """

#     # create a database connection
#     conn = create_connection(database)

#     # create tables
#     if conn is not None:

#         # create inmates table
#         create_table(conn, sql_create_inmates_table)

#     else:
#         print("Error! cannot create the database connection.")

# if __name__ == '__main__':
#     #create_connection(r"/Users/daminisharma/Dropbox/Harris MSCAPP/2019-20_Q3_Spring/Machine Learning/covid_decarceration/data/nc_doc.db")
#     main()
