"""Methods for extracting voxel betas from db_emorep.tbl_betas_sep_stim_gm.

get_subj_betas : return pd.DataFrame of single subject betas
GetBetas : get requested subject betas as np.array

"""

# %%
import os
import platform
import pandas as pd
import numpy as np
from typing import Tuple, Type
from contextlib import contextmanager
from multiprocessing import Pool
import mysql.connector

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
            os.environ["SQL_PASS_DING"]
        except KeyError as e:
            raise Exception(
                "No global variable 'SQL_PASS' defined in user env"
            ) from e

    def con_server(self):
        """Connect to MySQL server."""
        if "dcc" in platform.uname().node:
            self._connect_dcc()
        elif "labarserv2" in platform.uname().node:
            self._connect_labarserv2()

    def _connect_labarserv2(self):
        """Connect to MySQL server from labarserv2."""
        self.con = mysql.connector.connect(
            host="localhost",
            user=os.environ["USER"],
            password=os.environ["SQL_PASS_DING"],
            database="db_emorep",
        )

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
