import falcon
from tg_api import images

api = application = falcon.API()
images = images.Resource()
api.add_route('/images', images)
