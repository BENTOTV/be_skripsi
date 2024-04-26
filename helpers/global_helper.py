import jwt
from config import config
import config.response_handler as ResponseHandler

class GlobalHelper():

    def decode_token(token = ""):
        try:     
            token = token.split(" ")[1] if token.startswith("Bearer ") else token
            return jwt.decode(token, config.SECRET_KEY, algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            return ResponseHandler.unauthorized('Token has expired!')
        except jwt.InvalidTokenError:
            return ResponseHandler.unauthorized('Invalid token!')