import sqlite3
import pandas as pd


class DatabaseHandler:
    def __init__(self, db_name='housing_data.db'):
        self.conn = sqlite3.connect(db_name)

    def save_to_db(self, data, table_name):
        data.to_sql(table_name, con=self.conn, if_exists='replace', index=False)
        print(f"Data saved to {table_name} table in database.")

    def load_from_db(self, table_name):
        query = f"SELECT * FROM {table_name}"
        data = pd.read_sql(query, self.conn)
        print(f"Data loaded from {table_name} table.")
        return data

    def close_connection(self):
        self.conn.close()