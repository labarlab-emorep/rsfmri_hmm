# %%
import os
import mysql.connector
import pandas as pd

# %% establish a connection to the database
db_con_test = mysql.connector.connect(
    host="localhost", 
    user="yd169", 
    password="mango", 
    database="db_test_yd169"
)

# %% create a cursor object that has a number of methods available
# to interact with the mysql server
db_cur_test = db_con_test.cursor()


# %% select some data from the test database 
query = "select * from tbl_erq limit 10"

db_cur_test.execute(query)

# %% fetch the data and convert them into a pandas dataframe
result = db_cur_test.fetchall()
df_erq = pd.DataFrame(
    result, columns=["subj_id", "sess_id", "item_erq", "resp_erq"]
)


# %% remmeber to close the connection to the database
db_con_test.close()
