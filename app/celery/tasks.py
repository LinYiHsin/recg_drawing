import os
import subprocess
import time
import json
import datetime
import urllib

from app import create_worker

from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy import create_engine
from sqlalchemy.sql import text


from constants import *

from flask import Flask
from celery import Celery

flask_app = Flask(__name__)

flask_app.config['CELERY_BROKER_URL'] = CELERY_BROKER_URL
flask_app.config['CELERY_RESULT_BACKEND'] = CELERY_RESULT_BACKEND

worker = Celery(
    flask_app.name, 
    broker = flask_app.config['CELERY_BROKER_URL'],
    backend = flask_app.config['CELERY_RESULT_BACKEND']
)
# worker.conf.CELERY_TASK_TRACK_STARTED = True

# worker = create_worker()

@worker.task( bind=True )
def Hello( self ):

    print('task_id {}'.format(self.request.id))
    print(self.request)

    return None


def img2base64( img_path ):
    import io
    import base64
    from PIL import Image

    img = Image.open(img_path)
    imgByteArr = io.BytesIO()
    img.save(imgByteArr, format='PNG')
    img_value = imgByteArr.getvalue()

    return base64.b64encode(img_value).decode('ascii')


@worker.task( bind=True )
def Recognition( self, drawing_id, filename, drawing_path ):

    # connect to database
    # ========== database table ==========
    engine = create_engine( SQLALCHEMY_DATABASE_URI )
    Session = sessionmaker()
    Session.configure( bind = engine )
    session = Session()
    # ========== end of init database ==========

    task_id = self.request.id
    sql_cmd = f"INSERT INTO worker_task (created_on, updated_on, description, task_id) OUTPUT inserted.id VALUES(GETDATE(), GETDATE(), 'recognition', '{task_id}')"
    result = session.execute( text(sql_cmd) ).first()
    worker_id = result.id


    drawing_encoded = img2base64(drawing_path)

    # drawing -> views
    detect_path = os.path.join(YOLO_DIR_PATH, DETECT_PY)
    project_path = os.path.join(YOLO_DIR_PATH, 'run', 'detect')

    weight_path = os.path.join(YOLO_DIR_PATH, 'weights', drawing_views_weight)
    source_path = drawing_path
    project_name = f"{drawing_id}_{filename.rsplit('.', 1)[0]}"

    run_cmd = f'python {detect_path} --weight {weight_path} --img-size 640 --conf 0.5 --source {source_path} --project {project_path}  --name {project_name} --no-trace'
    subprocess.run(run_cmd)

    drawing_predic = os.path.join(project_path, project_name, filename)
    drawing_predic_encoded = img2base64(drawing_predic)

    sql_cmd = f"UPDATE drawings SET img='{drawing_encoded}', img_predict='{drawing_predic_encoded}', worker_id='{worker_id}' WHERE id={drawing_id}"
    session.execute( text(sql_cmd) )

    
    views_source_path = os.path.join(project_path, project_name, filename.rsplit('.', 1)[0].lower())
    view_files_list = next(os.walk(views_source_path), (None, None, []))[2]
    views_map = {}
    for view_file in view_files_list:
        view_path = os.path.join( views_source_path, view_file )
        view_encoded = img2base64(view_path)

        view_name = view_file.rsplit('.', 1)[0]
        views_map[ view_name ] = {}
        views_map[ view_name ]['view_encoded'] = view_encoded

    insert_cmd = "INSERT INTO views (created_on, updated_on, drawing_id, name, img, labeled, exported) OUTPUT inserted.id, inserted.name VALUES"
    values_cmd = ""
    isFirst = True
    for view_name in views_map:
        view = views_map[ view_name ]
        if isFirst:
            isFirst = False
        else:
            values_cmd += ','
        values_cmd += "(GETDATE(), GETDATE(), '{0}', '{1}', '{2}', 0, 0)".format( drawing_id, view_name, view['view_encoded'])
    sql_cmd = insert_cmd + values_cmd
    result = session.execute( text(sql_cmd) )

    for res in result:
        views_map[ res.name ]['id'] = res.id

    Logs.delay('info', 'Recongnize drawing -> views Completed.')


    # view -> DIM
    detect_path = os.path.join(YOLO_DIR_PATH, DETECT_PY)

    weight_path = os.path.join(YOLO_DIR_PATH, 'weights', view_DIMs_weight)
    source_path = views_source_path
    project_path = source_path
    project_name = "views_detect"

    run_cmd = f'python {detect_path} --weight {weight_path} --img-size 640 --conf 0.5 --source {source_path} --project {project_path}  --name {project_name} --no-trace --rotate-result'
    subprocess.run(run_cmd)

    views_prodict_dir_path = os.path.join(project_path, project_name)
    view_files_list = next(os.walk(views_prodict_dir_path), (None, None, []))[2]
    for view_file in view_files_list:
        view_path = os.path.join( views_prodict_dir_path, view_file )
        view_predict_encoded = img2base64(view_path)

        view_name = view_file.rsplit('.', 1)[0]
        views_map[ view_name ]['view_predict_encoded'] = view_predict_encoded


    img_predict_sql = ""
    views_id = []
    for name in views_map:
        view = views_map[name]
        views_id.append(view['id'])
        img_predict_sql += "WHEN id={} THEN '{}' ".format(view['id'], view['view_predict_encoded'])
    sql_cmd = "UPDATE views SET img_predict=(CASE {} END) WHERE id IN({})".format(img_predict_sql, ','.join(str(id) for id in views_id))
    session.execute( text(sql_cmd) )

    
    for view_name in views_map:
        view_id = views_map[view_name]['id']
        DIM_dir = view_name.rsplit('.', 1)[0].lower()
        project_path = os.path.join(views_prodict_dir_path, DIM_dir)
        # project_name = f'{DIM_dir}_detect'

        # DIMs image file
        DIM_source_path = os.path.join(project_path, 'DIM')
        if os.path.exists(DIM_source_path):
            DIM_files_list = next(os.walk(DIM_source_path), (None, None, []))[2]

            DIMs_map = {}
            for DIM_file in DIM_files_list:
                DIM_path = os.path.join( DIM_source_path, DIM_file )
                encoded = img2base64(DIM_path)

                DIM_name = DIM_file.rsplit('.', 1)[0]
                DIMs_map[ DIM_name ] = {}
                DIMs_map[ DIM_name ]['encoded'] = encoded

            # Dimsioning -> DIM type and recongnize
            detect_path = os.path.join(YOLO_DIR_PATH, DETECT_DIM_PY)
            weight_path = os.path.join(YOLO_DIR_PATH, 'weights', DIM_type_weight)
            project_name = os.path.join(project_path, 'DIM_detect')

            run_cmd = f'python {detect_path} --weight {weight_path} --img-size 200 --conf 0.5 --source {DIM_source_path} --project {project_path}  --name {project_name} --exist-ok --no-trace'
            subprocess.run(run_cmd)

            detect_dir_path = os.path.join( project_path, project_name )
            files_list = os.listdir(detect_dir_path)
            detect_img_files = [ img_file for img_file in files_list if not img_file.endswith('.json')] 
            for img in detect_img_files:
                img_path = os.path.join( detect_dir_path, img )
                img_encoded = img2base64(img_path)

                DIM_name = img.rsplit('.', 1)[0]
                DIMs_map[ DIM_name ]['predict_encoded'] = img_encoded

            json_path = os.path.join( detect_dir_path, 'recognize.json' )
            details_json = json.load(open(json_path))

            for detail in details_json:
                DIM_name = detail['image_name'].rsplit('.', 1)[0]
                datas = detail['datas']
                for data in datas:
                    if data['label'] == "dimension":
                        DIMs_map[ DIM_name ]["dimensioning"] = data['text']
                    elif data['label'] == "tolerance":
                        DIMs_map[ DIM_name ]["tolerance"] = data['text']
                    elif data['label'] == "tolerance_upper" or data['label'] == "tolerance_lower":
                        if 'tolerance' in DIMs_map[ DIM_name ]:
                            if data['label'] == "tolerance_upper":
                                DIMs_map[ DIM_name ]["tolerance"] = data['text'] + ', ' + DIMs_map[ DIM_name ]["tolerance"]
                            else:
                                DIMs_map[ DIM_name ]["tolerance"] = DIMs_map[ DIM_name ]["tolerance"] + ', ' + data['text']
                        else:
                            DIMs_map[ DIM_name ]["tolerance"] = data['text']



            if len(DIMs_map.keys()) != 0:
                insert_cmd = "INSERT INTO dimensionings (created_on, updated_on, view_id, name, img, img_predict, dimensioning, tolerance, measure, labeled, exported) VALUES"
                values_cmd = ""
                isFirst = True
                for DIM_name in DIMs_map:

                    DIM = DIMs_map[ DIM_name ]
                    if 'dimensioning' in DIM:
                        dimensioning = DIM['dimensioning']
                    else:
                        dimensioning = '--'

                    if 'tolerance' in DIM:
                        tolerance = DIM['tolerance']
                    else:
                        tolerance = '--'

                    if isFirst:
                        isFirst = False
                    else:
                        values_cmd += ','

                    values_cmd += "(GETDATE(), GETDATE(), '{0}', '{1}', '{2}', '{3}', '{4}', '{5}', 1, 0, 0)".format(view_id, DIM_name, DIM['encoded'], DIM['predict_encoded'], dimensioning, tolerance)

                sql_cmd = insert_cmd + values_cmd
                session.execute( text(sql_cmd) )


        # FCFs image file
        FCF_source_path = os.path.join(views_prodict_dir_path, DIM_dir, 'FCF')
        if os.path.exists(FCF_source_path):
            FCF_files_list = next(os.walk(FCF_source_path), (None, None, []))[2]

            subprocess.run( f'python { DETECT_FCF_PY } --path { FCF_source_path }' )
            json_path = os.path.join( FCF_source_path, 'recognize.json' )
            details_json = json.load(open(json_path))

            FCFs_map = {}
            for FCF_file in FCF_files_list:
                FCF_path = os.path.join( FCF_source_path, FCF_file )
                encoded = img2base64(FCF_path)

                FCF_name = FCF_file.rsplit('.', 1)[0]
                FCFs_map[ FCF_name ] = {}

                if FCF_name in details_json:
                    FCFs_map[ FCF_name ] = details_json[ FCF_name ]          
                FCFs_map[ FCF_name ]['encoded'] = encoded

            if len(FCFs_map.keys()) != 0:
                insert_cmd = "INSERT INTO fcfs (created_on, updated_on, view_id, name, img, symbol, tolerance, datum) VALUES"
                values_cmd = ""
                isFirst = True
                for FCF_name in FCFs_map:
                    FCF = FCFs_map[ FCF_name ]
                    symbol = FCF['symbol']
                    tolerance = FCF['stated_tolerance']
                    datum = '--'
                    if 'primary_datum' in FCF:
                        datum = FCF['primary_datum']
                    if 'secondary_datum' in FCF:
                        datum += f", {FCF['secondary_datum']}"
                    if 'tertiary_datum' in FCF:
                        datum += f", {FCF['tertiary_datum']}"

                    if isFirst:
                        isFirst = False
                    else:
                        values_cmd += ','

                    values_cmd += "(GETDATE(), GETDATE(), {0}, '{1}', '{2}', '{3}', '{4}', '{5}')".format(view_id, FCF_name, encoded, symbol, tolerance, datum)
                sql_cmd = insert_cmd + values_cmd
                session.execute( text(sql_cmd) )


        # Datums image file
        Datum_source_path = os.path.join(views_prodict_dir_path, DIM_dir, 'Datum')
        if os.path.exists(Datum_source_path):
            Datum_files_list = next(os.walk(Datum_source_path), (None, None, []))[2]

            subprocess.run( f'python { DETECT_DATUM_PY } --path { Datum_source_path }' )
            json_path = os.path.join( Datum_source_path, 'recognize.json' )
            details_json = json.load(open(json_path))

            Datums_map = {}
            for Datum_file in Datum_files_list:
                Datum_path = os.path.join( Datum_source_path, Datum_file )
                encoded = img2base64(Datum_path)

                Datum_name = Datum_file.rsplit('.', 1)[0]
                Datums_map[ Datum_name ] = {}

                if Datum_name in details_json:
                    Datums_map[ Datum_name ] = details_json[ Datum_name ]
                Datums_map[ Datum_name ]['encoded'] = encoded

            if len(Datums_map.keys()) != 0:
                insert_cmd = "INSERT INTO datums (created_on, updated_on, view_id, name, img, datum) VALUES"
                values_cmd = ""
                isFirst = True
                for Datum_name in Datums_map:
                    Datum = Datums_map[ Datum_name ]
                    datum = Datum['datum']
                    
                    if isFirst:
                        isFirst = False
                    else:
                        values_cmd += ','

                    values_cmd += "(GETDATE(), GETDATE(), {0}, '{1}', '{2}', '{3}')".format(view_id, Datum_name, encoded, datum)
                sql_cmd = insert_cmd + values_cmd
                session.execute( text(sql_cmd) )

        
        sql_cmd = "SELECT id, name FROM dimensionings WHERE view_id={}".format(view_id)
        result = session.execute( text(sql_cmd) )
        for res in result:
            DIMs_map[ res.name ]['id'] = res.id




    Logs.delay('info', 'Recongnize view -> DIM Completed.')


    Logs.delay('success', f'Recongnize { filename } Success.')

    session.commit()
    session.close()

    return None

@worker.task( bind=True )
def Logs( self, level, message ):

    # connect to database
    # ========== database table ==========
    engine = create_engine( SQLALCHEMY_DATABASE_URI )
    Session = sessionmaker()
    Session.configure( bind = engine )
    session = Session()
    # ========== end of init database ==========

    task_id = self.request.id
    sql_cmd = f"INSERT INTO worker_task (created_on, updated_on, description, task_id) VALUES(GETDATE(), GETDATE(), 'logs', '{task_id}')"

    sql_cmd = f"INSERT INTO logs ( created_on, updated_on, level, message ) VALUES (GETDATE(), GETDATE(), '{level}', '{message}')"
    session.execute( text(sql_cmd) )
    
    session.commit()
    session.close()
    return
