from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

from celery import Celery

import os
import constants

db = SQLAlchemy()

#  取得啟動文件資料夾路徑
# pjdir = os.path.abspath(os.path.dirname(__file__))

def create_worker():
    """
    Create a new Celery object and tie together the Celery config to the app's
    config. Wrap all tasks in the context of the application.

    :param app: Flask app
    :return: Celery app
    """
    app = Flask(__name__)
    app.config.from_object('app.settings')

    celery = Celery(
        app.import_name,
        broker = app.config['CELERY_BROKER_URL'],
        backend = app.config['CELERY_RESULT_BACKEND'],
        include = 'app.celery.tasks'
    )
    celery.conf.update( app.config )
    task_base = celery.Task

    class ContextTask(task_base):
        abstract = True

        def __call__(self, *args, **kwargs):
            with app.app_context():
                # return task_base.__call__(self, *args, **kwargs)
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery


def create_app():
    app = Flask(__name__)
    app.config.from_object('app.settings')
    CORS(app, resource = { r"/api/.*": { "origins": constants.CORS_ORIGINS } })

    db.init_app(app)

    #  很重要，一定要放這邊
    from app.views import main_blueprint

    app.register_blueprint( main_blueprint )

    return app