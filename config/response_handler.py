from http import HTTPStatus
from flask import jsonify


def get_success(data):
    return jsonify({
        "data": data,
        "meta": {
            "code": HTTPStatus.OK,
            "message": "Successfully get data",
            "status": "OK"
        }
    })


def create_success(data):
    return jsonify({
        "data": data,
        "meta": {
            "code": HTTPStatus.CREATED,
            "message": "Successfully create data",
            "status": "OK"
        }
    })

def custom_success_response(status, data, message):
    return jsonify({
        "data": data,
        "meta": {
            "code": status,
            "message": message,
            "status": "OK"
        }
    })

def bad_request(message):
    return jsonify({
        "data": "",
        "meta": {
            "code": HTTPStatus.BAD_REQUEST,
            "message": message,
            "status": "ERROR"
        }
    })

def unauthorized(message):
    return jsonify({
        "data": "",
        "meta": {
            "code": HTTPStatus.UNAUTHORIZED,
            "message": message,
            "status": "ERROR"
        }
    })

def custom_fail_response(status, data, message):
    return jsonify({
        "data": data,
        "meta": {
            "code": status,
            "message": message,
            "status": "ERROR"
        }
    })