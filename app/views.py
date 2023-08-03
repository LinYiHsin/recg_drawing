import os
import subprocess
import requests
import datetime
import urllib
from pdf2jpg import pdf2jpg
import base64
import shutil


from flask import Blueprint, render_template, jsonify, current_app, request, redirect, url_for
from flask import send_from_directory
from celery.result import AsyncResult

from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy import create_engine
from sqlalchemy.sql import text

from app.celery.tasks import *

from werkzeug.utils import secure_filename
from constants import DATETIME_FORMAT_STR, ALLOWED_EXTENSIONS, YOLO_DIR_PATH, drawing_views_weight, view_DIMs_weight

main_blueprint = Blueprint('Main', __name__)


@main_blueprint.route('/')
def hello():
    # Hello.delay('Name 1')
    return render_template('hello_world.html')


@main_blueprint.route('/overview')
def overview():

    return render_template('overview.html')


@main_blueprint.route('/labeled_overview')
def labeled_overview():

    return render_template('labeled_overview.html')


@main_blueprint.route('/drawing/views/<int:id>')
def drawing_views(id):

    return render_template('views.html', id = id)


@main_blueprint.route('/view/dimensionings/<int:id>')
def view_dimensionings(id):

    return render_template('dimensionings.html', id = id)


@main_blueprint.route('/label_image')
def label_image():

    stage = request.args.get('stage', default = 'drawing', type = str)
    image_id = request.args.get('image_id', type = int)
    image_name = request.args.get('image_name', type = str)

    return render_template('label_image.html', stage = stage, image_id = image_id, image_name = image_name)



@main_blueprint.route('/api/upload_file', methods=[ 'GET', 'POST' ])
def upload_file():
    
    if request.method == 'POST':

        file = request.files['file']
        filename = secure_filename(file.filename)
        extension = filename.rsplit('.', 1)[1]
        if file and (extension in ALLOWED_EXTENSIONS):

            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save( file_path )

            # convert PDF to JPEG
            if extension == 'pdf':
                result = pdf2jpg.convert_pdf2jpg( file_path, os.path.join(current_app.config['UPLOAD_FOLDER'], 'pdf2jpg'), pages='ALL' )
                page = 0
                file_path = result[0]['output_jpgfiles'][page]
                filename = '{}_{}.jpg'.format(page, filename)

            # connect to database
            # ========== database table ==========
            engine = create_engine( current_app.config['SQLALCHEMY_DATABASE_URI'] )
            Session = sessionmaker()
            Session.configure( bind = engine )
            session = Session()
            # ========== end of init database ==========

            sql_cmd = f"INSERT INTO drawings (created_on, updated_on, name, author, labeled, exported) OUTPUT inserted.id VALUES(GETDATE(), GETDATE(), '{ filename.rsplit('.', 1)[0] }', '{ request.remote_addr }', 0, 0)"
            result = session.execute( text(sql_cmd) ).first()
            drawing_id = result.id
            return_json = { 'id': drawing_id }

            Recognition.delay( drawing_id, filename, file_path )

            session.commit()
            session.close()

            Logs.delay('info', 'Drawing uploaded success.')

            return_json['success'] = 'Drawing uploaded success.'
        else:
            return_json = { 'error': 'Invalid file extension.' }

        return jsonify( return_json )

    else:
        return render_template('upload_file.html')


@main_blueprint.route('/api/edit_drawing', methods=[ 'POST', 'GET' ])
def edit_drawing():

    return_json={}

    if request.method == 'POST':

        # connect to database
        # ========== database table ==========
        engine = create_engine( current_app.config['SQLALCHEMY_DATABASE_URI'] )
        Session = sessionmaker()
        Session.configure( bind = engine )
        session = Session()
        # ========== end of init database ==========

        data = request.get_json(silent = True)
        if data['method'] == 'edit_name':
            sql_cmd = "UPDATE drawings SET updated_on=GETDATE(), name='{}' WHERE id={}".format(data['drawing_name'], data['drawing_id'])
            session.execute( text(sql_cmd) )

            return_json['success'] = 'Edit drawing name success.'

        elif data['method'] == 'delete_drawing':
            sql_cmd = "DELETE FROM drawings WHERE id={}".format(data['drawing_id'])
            session.execute( text(sql_cmd) )

            return_json['success'] = 'Delete drawing success.'

        
        session.commit()
        session.close()

    return jsonify( return_json )


@main_blueprint.route('/api/edit_annotation', methods=[ 'POST', 'GET' ])
def edit_annotation():

    return_json={}

    if request.method == 'POST':

        # connect to database
        # ========== database table ==========
        engine = create_engine( current_app.config['SQLALCHEMY_DATABASE_URI'] )
        Session = sessionmaker()
        Session.configure( bind = engine )
        session = Session()
        # ========== end of init database ==========

        data = request.get_json(silent = True)
        print(data)

        if data['method'] == 'edit':
            if data['annotation'] == 'dimensioning':
                sql_cmd = "UPDATE dimensionings SET updated_on=GETDATE(), dimensioning='{}', tolerance='{}' WHERE id={}".format(data['dimensioning'], data['tolerance'], data['id'])
            elif data['annotation'] == 'fcf':
                sql_cmd = "UPDATE fcfs SET updated_on=GETDATE(), symbol='{}', tolerance='{}', datum='{}' WHERE id={}".format(data['symbol'], data['tolerance'], data['datum'], data['id'])
            elif data['annotation'] == 'datum':
                sql_cmd = "UPDATE datums SET updated_on=GETDATE(), datum='{}' WHERE id={}".format(data['datum'], data['id'])
            session.execute( text(sql_cmd) )
            return_json['success'] = 'Edit success.'

        elif data['method'] == 'delete':
            if data['annotation'] == 'dimensioning':
                sql_cmd = "DELETE FROM dimensionings WHERE id={}".format(data['id'])
            elif data['annotation'] == 'fcf':
                sql_cmd = "DELETE FROM fcfs WHERE id={}".format(data['id'])
            elif data['annotation'] == 'datum':
                sql_cmd = "DELETE FROM datums WHERE id={}".format(data['id'])
            session.execute( text(sql_cmd) )
            return_json['success'] = 'Delete success.'

        
        session.commit()
        session.close()

    return jsonify( return_json )


@main_blueprint.route('/api/get_drawings')
def get_drawings():
    # connect to database
    # ========== database table ==========
    engine = create_engine( current_app.config['SQLALCHEMY_DATABASE_URI'] )
    Session = sessionmaker()
    Session.configure( bind = engine )
    session = Session()
    # ========== end of init database ==========

    return_json = []
    sql_cmd = "SELECT * FROM drawings"
    result = session.execute( text(sql_cmd) )
    for res in result:
        drawing = {}
        drawing['id'] = res.id
        drawing['drawing_name'] = res.name
        drawing['drawing_img'] = res.img
        if res.created_on:
            drawing['created_on'] = res.created_on.strftime(DATETIME_FORMAT_STR).rsplit('.', 1)[0]
        else:
            drawing['created_on'] = '----/--/-- --:--:--'

        return_json.append(drawing)

    session.commit()
    session.close()

    return jsonify( return_json )


@main_blueprint.route('/api/get_drawing_views/<int:drawing_id>', methods=[ 'GET' ])
def get_drawing_views(drawing_id):

    # connect to database
    # ========== database table ==========
    engine = create_engine( current_app.config['SQLALCHEMY_DATABASE_URI'] )
    Session = sessionmaker()
    Session.configure( bind = engine )
    session = Session()
    # ========== end of init database ==========

    return_json = {}
    sql_cmd = f"SELECT id, name, img, img_predict FROM drawings WHERE id = { drawing_id }"
    result = session.execute( text(sql_cmd) ).first()

    if result:

        if result.img_predict:
            return_json['id'] = result.id
            return_json['drawing_name'] = result.name
            return_json['drawing_img'] = result.img
            return_json['drawing_img_predict'] = result.img_predict
            return_json['views'] = []

            sql_cmd = f"SELECT id, name, img from views where drawing_id = { drawing_id }"
            result = session.execute( text(sql_cmd) )
            for res in result:
                view = {}
                view['id'] = res.id
                view['view_name'] = res.name
                view['view_img'] = res.img
                return_json['views'].append(view)

        else:
            return_json['error'] = "Recognition unfinished."

    else:
        return_json['error'] = f"Can't find drawing id {drawing_id}."


    session.commit()
    session.close()

    return jsonify( return_json )


@main_blueprint.route('/api/get_view_dimensionings/<int:view_id>', methods=[ 'GET' ])
def get_view_dimensionings(view_id):

    # connect to database
    # ========== database table ==========
    engine = create_engine( current_app.config['SQLALCHEMY_DATABASE_URI'] )
    Session = sessionmaker()
    Session.configure( bind = engine )
    session = Session()
    # ========== end of init database ==========

    return_json = {}
    sql_cmd="""SELECT drawings.name as drawing_name, drawings.id as drawing_id, views.id as view_id, views.name, views.img, views.img_predict FROM views 
                INNER JOIN drawings on drawings.id = views.drawing_id WHERE views.id = {}""".format(view_id)
    result = session.execute( text(sql_cmd) ).first()

    if result:

        if result.img_predict:
            return_json['drawing_name'] = result.drawing_name
            return_json['drawing_id'] = result.drawing_id
            return_json['id'] = result.view_id
            return_json['view_name'] = result.name
            return_json['view_img'] = result.img
            return_json['view_img_predict'] = result.img_predict
            return_json['dimensionings'] = []
            return_json['fcfs'] = []
            return_json['datums'] = []

            sql_cmd = "SELECT * from dimensionings where view_id = '{}'".format(view_id)
            result = session.execute( text(sql_cmd) )
            for res in result:
                dim = {}
                dim['id'] = res.id
                dim['dim_name'] = res.name
                dim['dim_img'] = res.img
                dim['dim_img_predict'] = res.img_predict
                dim['dimensioning'] = res.dimensioning
                dim['tolerance'] = res.tolerance
                dim['measure'] = res.measure
                return_json['dimensionings'].append(dim)

            sql_cmd = "SELECT * from fcfs where view_id = '{}'".format(view_id)
            result = session.execute( text(sql_cmd) )
            for res in result:
                fcf = {}
                fcf['id'] = res.id
                fcf['fcf_name'] = res.name
                fcf['fcf_img'] = res.img
                fcf['symbol'] = res.symbol
                fcf['tolerance'] = res.tolerance
                fcf['datum'] = res.datum
                return_json['fcfs'].append(fcf)

            sql_cmd = "SELECT * from datums where view_id = '{}'".format(view_id)
            result = session.execute( text(sql_cmd) )
            for res in result:
                datum = {}
                datum['id'] = res.id
                datum['datum_name'] = res.name
                datum['datum_img'] = res.img
                datum['datum'] = res.datum
                return_json['datums'].append(datum)

        else:
            return_json['error'] = "Recognition unfinished."

    else:
        return_json['error'] = f"Can't find view id {view_id}."


    session.commit()
    session.close()

    return jsonify( return_json )


@main_blueprint.route('/api/get_label_datas', methods=[ 'GET' ])
def get_label_datas():

    return_json = {}

    # connect to database
    # ========== database table ==========
    engine = create_engine( current_app.config['SQLALCHEMY_DATABASE_URI'] )
    Session = sessionmaker()
    Session.configure( bind = engine )
    session = Session()
    # ========== end of init database ==========

    stages = ['drawing', 'view', 'dimensioning']

    return_json[ 'label_datas' ] = []
    for stage in stages:
        if stage == 'drawing':
            table = 'drawings'
            label_table = 'drawing_labels'
            id_col = 'drawing_id'
        elif stage == 'view':
            table = 'views'
            label_table = 'view_labels'
            id_col = 'view_id'
        elif stage == 'dimensioning':
            table = 'dimensionings'
            label_table = 'dimensioning_labels'
            id_col = 'dimensioning_id'

        sql_cmd = """
            SELECT sub_table.created_on, {0}.id, {0}.name, {0}.img, {0}.exported  FROM
            (SELECT DISTINCT {2} FROM {1})AS A
            CROSS APPLY(SELECT TOP(1) * FROM {1} WHERE {2} = A.{2} ORDER BY created_on)sub_table
            INNER JOIN {0} ON {0}.id = sub_table.{2}
            ORDER BY sub_table.created_on DESC""".format(table, label_table, id_col)
        result = session.execute( text(sql_cmd) )

        for res in result:
            data = {
                'created_on': res.created_on.strftime(DATETIME_FORMAT_STR).rsplit('.', 1)[0],
                'id': res.id,
                'name': res.name,
                'img': res.img,
                'exported': res.exported,
                'stage': stage
            }
            return_json[ 'label_datas' ].append(data)


        
    session.commit()
    session.close()

    return jsonify( return_json )


@main_blueprint.route('/api/export_label_datas', methods=[ 'GET', 'POST' ])
def export_label_datas():

    now = datetime.datetime.now()
    current_time = now.strftime("%Y%m%d_%H%M%S")
    EXPORT_FOLDER = current_app.config['EXPORT_FOLDER']
    EXPORT_FOLDER = os.path.join(EXPORT_FOLDER, current_time)
    # print(EXPORT_FOLDER)
    return_json = {}

    # connect to database
    # ========== database table ==========
    engine = create_engine( current_app.config['SQLALCHEMY_DATABASE_URI'] )
    Session = sessionmaker()
    Session.configure( bind = engine )
    session = Session()
    # ========== end of init database ==========

    datas = request.get_json(silent = True)['datas']
    stages = ['drawing', 'view', 'dimensioning']
    for stage in stages:
        if stage == 'drawing':
            table = 'drawings'
            label_table = 'drawing_labels'
            id_col = 'drawing_id'
        elif stage == 'view':
            table = 'views'
            label_table = 'view_labels'
            id_col = 'view_id'
        elif stage == 'dimensioning':
            table = 'dimensionings'
            label_table = 'dimensioning_labels'
            id_col = 'dimensioning_id'

        filtered_datas = list(filter(lambda x: x['stage']==stage, datas))
        
        if len(filtered_datas) != 0:
            folder_path = os.path.join(EXPORT_FOLDER, stage)
            isExist = os.path.exists(folder_path)
            if not isExist:
                os.makedirs(os.path.join(folder_path, 'images'))
                os.makedirs(os.path.join(folder_path, 'labels'))

            export_datas = []
            ids = [ str(d['id']) for d in filtered_datas ]
            sql_cmd = f"SELECT id, name, img FROM {table} WHERE id IN ({', '.join(ids)})"
            img_result = session.execute( text(sql_cmd) ).fetchall()
            
            for res in img_result:
                id = res.id
                name = res.name
                filename = f'{id}_{name}'
                img = res.img
                img_decaded = base64.b64decode(img)
                img_file = open(os.path.join(folder_path, 'images', f'{filename}.png'), 'wb')
                img_file.write(img_decaded)
                img_file.close()

                sql_cmd = f"SELECT label_class, x_scale, y_scale, w_scale, h_scale FROM {label_table} WHERE {id_col}={id}"
                labels_result = session.execute( text(sql_cmd) ).fetchall()
                labels = []
                for label in labels_result:
                    labels.append(f"{label.label_class} {label.x_scale} {label.y_scale} {label.w_scale} {label.h_scale}\n")
                label_file = open(os.path.join(folder_path, 'labels', f'{filename}.txt'), 'w', encoding="utf-8")
                label_file.writelines(labels)
                label_file.close()

            sql_cmd = f"UPDATE {table} SET exported=1 WHERE id IN ({', '.join(ids)})"
            session.execute( text(sql_cmd) )

    
    session.commit()
    session.close()

    shutil.make_archive(EXPORT_FOLDER, 'zip', EXPORT_FOLDER)
    zip_filename = f"{current_time}.zip"

    return send_from_directory( directory=current_app.config['EXPORT_FOLDER'], path=zip_filename, as_attachment=True )


@main_blueprint.route('/api/new_label_data/<stage>/<image_id>', methods=[ 'GET', 'POST' ])
def new_label_data(stage, image_id):

    return_json = {}

    if stage == 'drawing':
        return_json['labels'] = [ 'View' ]
        table_name = 'drawings'
        label_table = 'drawing_labels'
        id_col = 'drawing_id'
    elif stage == 'view':
        return_json['labels'] = [ 'DIM', 'FCF', 'Datum' ]
        table_name = 'views'
        label_table = 'view_labels'
        id_col = 'view_id'
    elif stage == 'dimensioning':
        return_json['labels'] = [ 'Dimension', 'Tolerance Upper', 'Tolerance Lower', 'Tolerance' ]
        table_name = 'dimensionings'
        label_table = 'dimensioning_labels'
        id_col = 'dimensioning_id'
    else:
        return jsonify( {'error' : 'Wrong stage'} )

    # connect to database
    # ========== database table ==========
    engine = create_engine( current_app.config['SQLALCHEMY_DATABASE_URI'] )
    Session = sessionmaker()
    Session.configure( bind = engine )
    session = Session()
    # ========== end of init database ==========

    if request.method == 'POST':
        datas = request.get_json(silent = True)
        image_width = datas['image_width']
        image_height = datas['image_height']

        session.execute( text(f"DELETE FROM {label_table} WHERE {id_col}={image_id}") )
        insert_cmd = f"INSERT INTO {label_table} VALUES"
        values_cmd = ""
        isFirst = True
        for data in datas['labels_data']:

            x_scale = (data['x'] + data['width']/2) / image_width
            y_scale = (data['y'] + data['height']/2) / image_height
            w_scale = abs(data['width']) / image_width
            h_scale = abs(data['height']) / image_height

            if isFirst:
                isFirst = False
            else:
                values_cmd += ","
            values_cmd += f"(GETDATE(), GETDATE(), {image_id}, {data['class']}, {x_scale:.6f}, {y_scale:.6f}, {w_scale:.6f}, {h_scale:.6f})"
        
        sql_cmd = insert_cmd + values_cmd
        session.execute( text(sql_cmd) )
        session.execute( text(f"UPDATE {table_name} SET labeled=1, exported=0 WHERE id={image_id}") )
        return_json = {'success': 'Label datas saved success.'}

    else:

        sql_cmd = f"SELECT id, img, labeled FROM {table_name} WHERE id={image_id}"
        result = session.execute( text(sql_cmd) ).first()

        return_json['img'] = result.img

        return_json['labels_data'] = []

        if result.labeled:
            sql_cmd = f"SELECT * FROM {label_table} WHERE {id_col}={image_id}"
            result = session.execute( text(sql_cmd) )
            for res in result:
                return_json['labels_data'].append({
                    'x_scale': res.x_scale,
                    'y_scale': res.y_scale,
                    'w_scale': res.w_scale,
                    'h_scale': res.h_scale,
                    'label_class': res.label_class
                })


    session.commit()
    session.close()

    return jsonify( return_json )


@main_blueprint.route('/api/get_message', methods=[ 'GET' ])
def get_message():

    # connect to database
    # ========== database table ==========
    engine = create_engine( current_app.config['SQLALCHEMY_DATABASE_URI'] )
    Session = sessionmaker()
    Session.configure( bind = engine )
    session = Session()
    # ========== end of init database ==========

    return_json = {}
    sql_cmd = "SELECT * FROM logs WHERE created_on > DATEADD(mi, -1, GETDATE()) ORDER BY created_on DESC"
    
    result = session.execute( text(sql_cmd) ).first()
    if result:
        return_json = {
            'created_on': result.created_on.strftime(DATETIME_FORMAT_STR).rsplit('.', 1)[0],
            'level': result.level,
            'message': result.message
        }

    else:
        return_json = { 'empty': '' }


    session.commit()
    session.close()

    return jsonify( return_json )


@main_blueprint.route('/api/check_task_state/<task_id>', methods=[ 'GET' ])
def check_task_state(task, task_id):

    if task == 'recognition':
        return Recognition.AsyncResult(task_id).state

    return None