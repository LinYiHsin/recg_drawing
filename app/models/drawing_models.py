from app import db
import datetime


class WorkerTask( db.Model ):
    __tablename__ = 'worker_task'
    created_on = db.Column( db.DateTime, default=db.func.now() )
    updated_on = db.Column( db.DateTime, default=db.func.now() )
    id = db.Column( db.Integer, primary_key = True )
    description = db.Column( db.String )
    task_id = db.Column( db.String )

class Drawing( db.Model ):
    __tablename__ = 'drawings'
    created_on = db.Column( db.DateTime, default=db.func.now() )
    updated_on = db.Column( db.DateTime, default=db.func.now() )
    id = db.Column( db.Integer, primary_key = True )
    name = db.Column( db.String )
    img = db.Column( db.String )
    img_predict = db.Column( db.String )
    worker_id = db.Column( db.Integer, db.ForeignKey('worker_task.id', ondelete='CASCADE') )
    author = db.Column( db.String )
    labeled = db.Column( db.Boolean, nullable = False, default = False )
    exported = db.Column( db.Boolean, nullable = False, default = False )

class View( db.Model ):
    __tablename__ = 'views'
    created_on = db.Column( db.DateTime, default=db.func.now() )
    updated_on = db.Column( db.DateTime, default=db.func.now() )
    id = db.Column( db.Integer, primary_key = True )
    drawing_id = db.Column( db.Integer, db.ForeignKey('drawings.id', ondelete='CASCADE'), nullable = False )
    name = db.Column( db.String )
    img = db.Column( db.String )
    img_predict = db.Column( db.String )
    labeled = db.Column( db.Boolean, nullable = False, default = False )
    exported = db.Column( db.Boolean, nullable = False, default = False )

class Dimensioning( db.Model ):
    __tablename__ = 'dimensionings'
    created_on = db.Column( db.DateTime, default=db.func.now() )
    updated_on = db.Column( db.DateTime, default=db.func.now() )
    id = db.Column( db.Integer, primary_key = True )
    view_id = db.Column( db.Integer, db.ForeignKey('views.id', ondelete='CASCADE'), nullable = False )
    name = db.Column( db.String )
    img = db.Column( db.String )
    img_predict = db.Column( db.String )
    dimensioning = db.Column( db.String )
    tolerance = db.Column( db.String )
    measurement_value = db.Column( db.String )
    measure = db.Column( db.Boolean, nullable = False, default = False )
    labeled = db.Column( db.Boolean, nullable = False, default = False )
    exported = db.Column( db.Boolean, nullable = False, default = False )

class FCF( db.Model ):
    __tablename__ = 'fcfs'
    created_on = db.Column( db.DateTime, default=db.func.now() )
    updated_on = db.Column( db.DateTime, default=db.func.now() )
    id = db.Column( db.Integer, primary_key = True )
    view_id = db.Column( db.Integer, db.ForeignKey('views.id', ondelete='CASCADE'), nullable = False )
    name = db.Column( db.String )
    img = db.Column( db.String )
    symbol = db.Column( db.String )
    tolerance = db.Column( db.String )
    datum = db.Column( db.String )

class Datum( db.Model ):
    __tablename__ = 'datums'
    created_on = db.Column( db.DateTime, default=db.func.now() )
    updated_on = db.Column( db.DateTime, default=db.func.now() )
    id = db.Column( db.Integer, primary_key = True )
    view_id = db.Column( db.Integer, db.ForeignKey('views.id', ondelete='CASCADE'), nullable = False )
    name = db.Column( db.String )
    img = db.Column( db.String )
    datum = db.Column( db.String )


class DrawingLabel( db.Model ):
    __tablename__ = 'drawing_labels'
    created_on = db.Column( db.DateTime, default=db.func.now() )
    updated_on = db.Column( db.DateTime, default=db.func.now() )
    id = db.Column( db.Integer, primary_key = True )
    drawing_id = db.Column( db.Integer, db.ForeignKey('drawings.id', ondelete='CASCADE'), nullable = False )
    label_class = db.Column( db.Integer )
    x_scale = db.Column( db.Float )
    y_scale = db.Column( db.Float )
    w_scale = db.Column( db.Float )
    h_scale = db.Column( db.Float )

class ViewLabel( db.Model ):
    __tablename__ = 'view_labels'
    created_on = db.Column( db.DateTime, default=db.func.now() )
    updated_on = db.Column( db.DateTime, default=db.func.now() )
    id = db.Column( db.Integer, primary_key = True )
    view_id = db.Column( db.Integer, db.ForeignKey('views.id', ondelete='CASCADE'), nullable = False )
    label_class = db.Column( db.Integer )
    x_scale = db.Column( db.Float )
    y_scale = db.Column( db.Float )
    w_scale = db.Column( db.Float )
    h_scale = db.Column( db.Float )

class DimensioningLabel( db.Model ):
    __tablename__ = 'dimensioning_labels'
    created_on = db.Column( db.DateTime, default=db.func.now() )
    updated_on = db.Column( db.DateTime, default=db.func.now() )
    id = db.Column( db.Integer, primary_key = True )
    dimensioning_id = db.Column( db.Integer, db.ForeignKey('dimensionings.id', ondelete='CASCADE'), nullable = False )
    label_class = db.Column( db.Integer )
    x_scale = db.Column( db.Float )
    y_scale = db.Column( db.Float )
    w_scale = db.Column( db.Float )
    h_scale = db.Column( db.Float )

class Log( db.Model ):
    __tablename__ = 'logs'
    created_on = db.Column( db.DateTime, default=db.func.now() )
    updated_on = db.Column( db.DateTime, default=db.func.now() )
    id = db.Column( db.Integer, primary_key = True )
    level = db.Column( db.String )
    message = db.Column( db.String )
