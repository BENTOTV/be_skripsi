from flask import Blueprint, request, jsonify
from config.config import Config
from middlewares.auth_middleware import token_required
from models.user import User
import jwt
from http import HTTPStatus
import config.response_handler as ResponseHandler
from helpers.global_helper import GlobalHelper

auth_blueprint = Blueprint('auth', __name__)
config = Config()

@auth_blueprint.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data['username']
    password = data['password']
    User.create(username, password)
    return ResponseHandler.create_success("")

@auth_blueprint.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data['username']
    password = data['password']
    user = User.find_by_username(username)

    if user and user[2] == password:

        token = jwt.encode(
            {
                'id': user[0],
                'username': username
            }, 
            config.SECRET_KEY, 
            algorithm='HS256'    
        )
        
        return ResponseHandler.custom_success_response(status=HTTPStatus.OK, data={'token': token}, message="Login successful!")
    else:
        return ResponseHandler.custom_success_response(status=HTTPStatus.BAD_REQUEST, data="", message="Invalid username or password!")
    
    
@auth_blueprint.route('/test-protected', methods=['GET'])
@token_required
def protected(current_user):
    return ResponseHandler.get_success("Kalau saya muncul berarti kamu sudah login :)")
