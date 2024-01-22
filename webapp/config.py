import os
basedir = os.path.abspath(os.path.dirname(__file__))


def get_database_uri(env='development') -> str:
    if env == 'development':
        username: str = os.environ['POSTGRES_USERNAME']
        password: str = os.environ['POSTGRES_PASSWORD']
        host: str = 'localhost'
        port: int = 5432
    else:
        raise Exception('Not supported yet.')

    return f'postgresql://{username}:{password}@{host}:{port}/tornado'


class Config(object):
    ENV: str = 'development'
    DEBUG: bool = True
    TESTING: bool = False
    MAX_SHOW: int = 25
    # Uncomment me for actual version...
    # SQLALCHEMY_DATABASE_URI: str = get_database_uri(env = 'development')
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False
