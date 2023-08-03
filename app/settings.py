# Settings common to all environments (development|staging|production)
# Place environment specific settings in env_settings.py
# An example file (env_settings_example.py) can be used as a starting point
import environ
import urllib
from constants import databaseConfig, SQLALCHEMY_DATABASE_URI, UPLOAD_DIR_PATH, EXPORT_DIR_PATH, CELERY_BROKER_URL, CELERY_RESULT_BACKEND

# Application settings
APP_NAME = "Drawings Recognization"
APP_SYSTEM_ERROR_SUBJECT_LINE = APP_NAME + " system error"

# Flask-SQLAlchemy settings
SQLALCHEMY_TRACK_MODIFICATIONS = False

# *****************************
# Environment specific settings
# *****************************

# MARK OUT in production <========================== IMPORTANT!!!!
DEBUG = True

# DO NOT use Unsecure Secrets in production environments
# Generate a safe one with:
#     python -c "import os; print repr(os.urandom(24));"
SECRET_KEY = 'SECRET_KEY'

# SQLAlchemy settings
SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI
SQLALCHEMY_TRACK_MODIFICATIONS = False    # Avoids a SQLAlchemy Warning

# upload file
UPLOAD_FOLDER = UPLOAD_DIR_PATH
EXPORT_FOLDER = EXPORT_DIR_PATH
MAX_CONTENT_LENGTH = 20*1024*1024 # 20 megabytes

# celery
CELERY_BROKER_URL = CELERY_BROKER_URL
CELERY_RESULT_BACKEND = CELERY_RESULT_BACKEND
