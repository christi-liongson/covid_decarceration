import sqlite3
import pandas as pd
from sqlite3 import Error

import config
from create_db import create_connection, create_table, clean_column_names

def query_db(conn):
    '''
    Function to query database and save results as pandas dataframe for further analysis and cleaning
    '''

    try:
        c = conn.cursor()
        sql = 'SELECT * FROM OFNT3CE1 limit 1000;'

        df = pd.read_sql_query(sql,conn)
        #c.execute(sql)


    except Error as e:
        print(e)

    return(df)

# For testing
# def main():

#     conn = create_connection(config.database_name)
#     df = query_db(conn)
#     print(df)

# if __name__ == '__main__':
#     main()