from flask_script import Command
from app import db
from app import create_app
from app.models.drawing_models import Drawing, View, Dimensioning, Log

app = create_app()
db.app = app

class InitDbCommand( Command ):
    """ Initialize the database."""

    def run(self):
        init_db()

        return None

def init_db():
    """ Initialize the database."""
    db.drop_all()
    db.create_all()

    print("DB ready.")

    return None
