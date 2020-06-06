import sqlite3
import pandas as pd
from sqlite3 import Error

import config
from create_db import create_connection, create_table, clean_column_names

# def query_db(conn):
#     '''
#     Function to query database and save results as pandas dataframe for further analysis and cleaning
#     '''

#     try:
#         c = conn.cursor()
#         sql = '''SELECT * FROM INMT4AA1
#                 LEFT JOIN OFNT3CE1
#                     ON OFNT3CE1.OFFENDER_NC_DOC_ID_NUMBER = INMT4AA1.INMATE_DOC_NUMBER
#                 limit 10500;'''
#         sql = '''SELECT * FROM OFNT3CE1 WHERE OFFENDER_NC_DOC_ID_NUMBER = '0130556';'''
#         df = pd.read_sql_query(sql,conn)
#         #c.execute(sql)


#     except Error as e:
#         print(e)

#     return(df)

def query_db(conn,sql):
    '''
    Function to query database and save results as pandas dataframe for further analysis and cleaning
    '''

    try:
        c = conn.cursor()
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