from flask_script import Manager, Server
# from OpenSSL  import SSL

from app import create_app
from app.init.init_db import InitDbCommand


manager = Manager( create_app )
manager.add_command('init_db', InitDbCommand )
manager.add_command('runserver', Server( host='0.0.0.0', port=443, threaded=True ))
# manager.add_command('runserver', Server( host='0.0.0.0', port=443, ssl_crt='app/ssl/server.crt', ssl_key='app/ssl/server.key', threaded=True ))

if __name__ == "__main__":
    manager.run()