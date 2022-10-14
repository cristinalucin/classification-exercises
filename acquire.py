import os

import pandas as pd
import numpy as np

import env

def get_titanic_data():
    filename = "titanic.csv"
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        url = env.get_db_url('titanic_db')
        return pd.read_sql('SELECT * FROM passengers', url)
    
titanic_df = get_titanic_data()
titanic_df.to_csv("titanic.csv")

def get_iris_data():
    filename = "iris.csv"
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        from env import user, password, host
        url = f'mysql+pymysql://{user}:{password}@{host}/iris_db'
        url = env.get_db_url('iris_db')
        return pd.read_sql('SELECT * FROM measurements JOIN iris_db.species USING(species_id)', url)

iris_df = get_iris_data()
iris_df.to_csv("iris.csv")


def get_telco_data():
    filename = "telco.csv"
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        from env import user, password, host
        url = f'mysql+pymysql://{user}:{password}@{host}/telco_churn'
        url = env.get_db_url('telco_churn')
        return pd.read_sql('SELECT * FROM customers\
        JOIN telco_churn.contract_types USING(contract_type_id)\
        JOIN telco_churn.internet_service_types USING(internet_service_type_id)\
        JOIN telco_churn.payment_types USING(payment_type_id)', url)

telco_df = get_telco_data()
telco_df.to_csv("telco.csv")
    
