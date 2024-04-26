from functools import wraps
from flask import request
import jwt
from config.config import Config
import config.response_handler as ResponseHandler

config = Config()

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return ResponseHandler.unauthorized('Token is missing!')

        try:
            token = token.split(" ")[1] if token.startswith("Bearer ") else token
            data = jwt.decode(token, config.SECRET_KEY, algorithms=['HS256'])
            kwargs['current_user'] = data['username']
            return f(*args, **kwargs)
        except jwt.ExpiredSignatureError:
            return ResponseHandler.unauthorized('Token has expired!')
        except jwt.InvalidTokenError:
            return ResponseHandler.unauthorized('Invalid token!')

    return decorated
