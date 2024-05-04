from flask import Blueprint, request, jsonify
from config.config import Config
from middlewares.auth_middleware import token_required
from models.task import Task
import jwt
from http import HTTPStatus
import config.response_handler as ResponseHandler
from helpers.global_helper import GlobalHelper

task_blueprint = Blueprint('task', __name__)
config = Config()

@task_blueprint.route('/latihan', methods=['POST'])
def latihan():
    data = request.get_json()
    username = data['username']
    index = data['index']
    value = data['value']
    Task.inputResultlatihan( username, index, value)
    return ResponseHandler.create_success("")


@task_blueprint.route('/ujian', methods=['POST'])
def ujian():
    data = request.get_json()
    username = data['username']
    index = data['index']
    value = data['value']
    Task.inputResultujian( username, index, value)
    return ResponseHandler.create_success("")

@task_blueprint.route('/getlatihan', methods=['POST'])
def getLatihan():
    data = request.get_json()
   
    index = data['index']
    task = Task.find_latihan_by_index(index)

  
        
    return ResponseHandler.custom_success_response(status=HTTPStatus.OK, data={'data': task}, message="Data Latihan Get!")
 
 
@task_blueprint.route('/getujian', methods=['POST'])
def getLatihan():
    data = request.get_json()
   
    index = data['index']
    task = Task.find_ujian_by_index(index)

  
        
    return ResponseHandler.custom_success_response(status=HTTPStatus.OK, data={'data': task}, message="Data ujian Get!")
 
    
    
@task_blueprint.route('/test-protected', methods=['GET'])
@token_required
def protected(current_user):
    return ResponseHandler.get_success("Kalau saya muncul berarti kamu sudah login :)")
