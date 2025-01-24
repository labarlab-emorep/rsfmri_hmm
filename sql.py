import os
import platform
import pandas as pd
import numpy as np
from typing import Tuple, Type
from contextlib import contextmanager
from multiprocessing import Pool
import mysql.connector
import pymysql
import paramiko
from sshtunnel import SSHTunnelForwarder

rsa_ls2 = paramiko.RSAKey.from_private_key_file(os.environ["RSA_KEY"])

# %%
class _DbConnect:
    """Supply db_emorep database connection and cursor.

    Attributes
    ----------
    con : mysql.connector.connection_cext.CMySQLConnection
        Connection object to database

    Methods
    -------
    con_close()
        Close database connection
    con_cursor()
        Yield cursor object
    con_server()
        Connect to sql server

    Notes
    -----
    - Environmental variable 'SQL_PASS' containing mysql password required
    - Environmental variable 'RSA_LS2' containing path to RSA key
        for labarserv2 required on DCC

    Example
    -------
    db_con = _DbConnect()
    db_con.con_server()
    with db_con.con_cursor() as cur:
        cur.execute("select * from tbl_betas_sep_stim_gm limit 10")
        rows = cur.fetchall()
    db_con.con_close()

    """

    def __init__(self):
        """Initialize."""
        try:
            os.environ["SQL_PASS"]
        except KeyError as e:
            raise Exception(
                "No global variable 'SQL_PASS' defined in user env"
            ) from e

    def _establish_ssh(self):
        ssh_tunnel = SSHTunnelForwarder(
            ("ccn-labarserv2.vm.duke.edu", 22),
            ssh_username="ajt63",
            ssh_pkey=rsa_ls2,
            remote_bind_address=("127.0.0.1", 3306),
        )
        ssh_tunnel.start()

        self.con = pymysql.connect(
            host="127.0.0.1",
            user="ajt63",
            passwd=os.environ["SQL_PASS"],
            db="db_emorep",
            port=ssh_tunnel.local_bind_port,
        )
    
    def con_server(self):
        """Connect to MySQL server."""
        #if "dcc" in platform.uname().node:
            #self._connect_dcc()
        #if "labarserv2" in platform.uname().node:
        #self._connect_labarserv2()
        self._establish_ssh()

    def _connect_labarserv2(self):
        """Connect to MySQL server from labarserv2.
        self.con = mysql.connector.connect(
            host="localhost",
            user="ajt63",
            password='mango',
            database="db_emorep",
        ) """

        


    @contextmanager
    def con_cursor(self):
        """Yield cursor."""
        cursor = self.con.cursor()
        try:
            yield cursor
        finally:
            cursor.close()

    def con_close(self):
        """Close database connection."""
        self.con.close()