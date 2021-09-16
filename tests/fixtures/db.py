from dacapo.store import MongoConfigStore, MongoStatsStore

import pymongo
import pytest
from dotenv import load_dotenv

import os

def db_available():
    load_dotenv()
    db_host = os.getenv("mongo_db_host")
    db_name = os.getenv("mongo_db_name")
    client = pymongo.MongoClient(db_host, db_name)
    try:
        client.admin.command('ismaster')
        return True
    except pymongo.errors.ConnectionFailure:
        return False


@pytest.fixture
def mongo_config_store():
    load_dotenv()
    db_host = os.getenv("mongo_db_host")
    db_name = os.getenv("mongo_db_name")
    config_store = MongoConfigStore(db_host, db_name)
    yield config_store

@pytest.fixture
def mongo_stats_store():
    load_dotenv()
    db_host = os.getenv("mongo_db_host")
    db_name = os.getenv("mongo_db_name")
    config_store = MongoStatsStore(db_host, db_name)
    yield config_store