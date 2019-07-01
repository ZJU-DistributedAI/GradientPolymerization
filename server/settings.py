import os
from tornado.options import define, options

#env/.env
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'env/.env')
if os.path.exists(env_path):
    from decouple import Config, RepositoryEnv
    config = Config(RepositoryEnv(env_path))
else:
    from decouple import config

DEBUG = config('DEBUG', default=True, cast=bool)
define("polymerization_container", default=config('polymerization_container', default='127.0.0.1:8888'))
