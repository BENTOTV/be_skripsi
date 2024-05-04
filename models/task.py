import pymysql
from flask import current_app

def get_db():
    return pymysql.connect(
        host=current_app.config['MYSQL_HOST'],
        user=current_app.config['MYSQL_USER'],
        password=current_app.config['MYSQL_PASSWORD'],
        database=current_app.config['MYSQL_DB']
    )

class Task:
    @staticmethod
    def inputResultlatihan(username, index, result):
        connection = get_db()
        with connection.cursor() as cursor:
            cursor.execute("INSERT INTO latihan (username, index, value) VALUES (%s, %s,%s)", (username, index, result))
            connection.commit()
        connection.close()
        
    @staticmethod    
    def inputResultujian(username, index, result):
        connection = get_db()
        with connection.cursor() as cursor:
            cursor.execute("INSERT INTO ujian (username, index, value) VALUES (%s, %s,%s)", (username, index, result))
            connection.commit()
        connection.close()

    @staticmethod
    def find_latihan_by_index(index):
        connection = get_db()
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM latihan WHERE index = %s", (index,))
            task = cursor.fetchone()
        connection.close()
        return task
    
    @staticmethod
    def find_ujian_by_index(index):
        connection = get_db()
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM ujian WHERE index = %s", (index,))
            task1 = cursor.fetchone()
        connection.close()
        return task1
