from dacapo.store import MongoConfigStore, MongoStatsStore

import pymongo
import pytest
from dotenv import load_dotenv

import os


def db_available():
    db_host = os.getenv("MONGO_DB_HOST")
    client = pymongo.MongoClient(db_host, serverSelectionTimeoutMS=1000)
    try:
        client.admin.command("ping")
        return True
    except pymongo.errors.ConnectionFailure:
        return False


@pytest.fixture
def mongo_config_store():
    load_dotenv()
    db_host = os.getenv("MONGO_DB_HOST")
    db_name = os.getenv("MONGO_DB_NAME")
    config_store = MongoConfigStore(db_host, db_name)
    yield config_store
    client = pymongo.MongoClient(db_host)
    client.drop_database(db_name)


@pytest.fixture
def mongo_stats_store():
    load_dotenv()
    db_host = os.getenv("MONGO_DB_HOST")
    db_name = os.getenv("MONGO_DB_NAME")
    config_store = MongoStatsStore(db_host, db_name)
    yield config_store
    client = pymongo.MongoClient(db_host)
    client.drop_database(db_name)


load_dotenv()
DB_AVAILABLE = db_available()
