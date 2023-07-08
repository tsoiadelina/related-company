import pandas as pd
import sqlite3


class DBConnection:
    def __init__(self):

        self.conn = None
        try:
            self.conn = sqlite3.connect('db_okved')
        except Exception as e:
            print(e)

    def insert(self, row: tuple[str]):
        sql = '''
        insert into main.requests_to_okved (dt_req, okved, num_outs, first_out)
        values (?,?,?,?) 
        '''
        cur = self.conn.cursor()
        cur.execute(sql, row)
        self.conn.commit()

