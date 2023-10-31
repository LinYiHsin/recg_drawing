import os
import urllib

# MSSQL
databaseConfig = {
    'driver': '{ODBC Driver 17 for SQL Server}',
    'host': '.xxx.xxx.xxx.xxx',
    'username': 'USERNAME',
    'password': 'PASSWORD',
    'database': 'DATABASE',
}
sql_params = urllib.parse.quote_plus("DRIVER={0};SERVER={1};DATABASE={2};UID={3};PWD={4}".format( databaseConfig['driver'], databaseConfig['host'], databaseConfig['database'], databaseConfig['username'], databaseConfig['password'] ))
SQLALCHEMY_DATABASE_URI = "mssql+pyodbc:///?odbc_connect=%s" % sql_params


# Flask
CORS_ORIGINS = []


DATETIME_FORMAT_STR = '%Y-%m-%d %H:%M:%S.%f'


ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg'])
UPLOAD_DIR_PATH = os.path.join(os.getcwd(), 'uploadFiles')
EXPORT_DIR_PATH = os.path.join(os.getcwd(), 'exportFiles')


# yolov7
YOLO_DIR_PATH = os.path.join(os.getcwd(), 'yolov7')
DETECT_PY = "detect_view.py"
DETECT_DIM_PY = "detect_dim.py"
drawing_views_weight = '0519_view_best.pt'
view_DIMs_weight = '1013_DIMandFCF_best.pt'
DIM_type_weight = '0324_dim_and_tol_best.pt'

# subprocess path
DETECT_FCF_PY = os.path.join(os.getcwd(), 'app', 'subprocess', 'detect_fcf.py')
DETECT_DATUM_PY = os.path.join(os.getcwd(), 'app', 'subprocess', 'detect_datum.py')


# celery
CELERY_BROKER_URL = 'redis://127.0.0.1:6379/0'
CELERY_RESULT_BACKEND = 'redis://127.0.0.1:6379/0'
