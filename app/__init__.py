from flask import Flask

def create_app():
    app = Flask(__name__)

    from config import config
    app.config.from_object(config)

    from controllers.auth_controller import auth_blueprint
    app.register_blueprint(auth_blueprint)
    
    # from controllers. import task_blueprint
    # app.register_blueprint(task_blueprint)
    
    from controllers.mfcc import mfcc_blueprint
    app.register_blueprint(mfcc_blueprint)

    return app
