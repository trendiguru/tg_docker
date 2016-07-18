from bson import json_util
from rq import Queue

import falcon
from . import page_results
from . import constants

q1 = Queue('new_images', connection=constants.redis_conn)


class Resource(object):
    cors_headers = {'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Credentials': 'true',
                    'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE',
                    'Access-Control-Max-Age': '86400'}

    def on_get(self, req, resp):
        ret = {}
        page_url = req.get_param("pageUrl")
        image_url = req.get_param("imageUrl")
        exists = req.get_param("exists")
        relevant = req.get_param("relevant")
        lang = req.get_param("lang")

        if 'fashionseoul' in image_url:
            products = "GangnamStyle"
        else:
            products = "ShopStyle"

        if lang:
            page_results.set_lang(lang)

        if page_url:
            ret = page_results.search_existing_images(page_url)
        elif image_url:
            if exists:
                ret = {"exists": page_results.image_exists(image_url)}
            elif relevant:
                ret = {"relevant": page_results.is_image_relevant(image_url)}
            else:
                ret = page_results.get_data_for_specific_image(image_url=image_url, products_collection=products)
        else:
            resp.status = falcon.HTTP_400

        # ret = {"hello": "world!",
        #         "collection_names": constants.db.collection_names()}

        resp.data = json_util.dumps(ret) + "\n"
        resp.content_type = 'application/json'
        resp.set_headers(self.cors_headers)
        resp.status = falcon.HTTP_200

    def on_post(self, req, resp):
        ret = {"success": False}
        try:
            data = json_util.loads(req.stream.read())
            page_url = data.get("pageUrl")
            images = data.get("imageList")
            if 'method' in req.params:
                method = req.params["method"]
            else:
                method = 'pd'
            if type(images) is list and page_url is not None:
                relevancy_dict = {url: page_results.route_by_url(url, page_url, method) for url in images}
                ret["success"] = True
                ret["relevancy_dict"] = relevancy_dict
            else:
                ret["success"] = False
                ret["error"] = "Missing image list and/or page url"

        except Exception as e:
            ret["error"] = str(e)
        
        resp.data = json_util.dumps(ret)
        resp.content_type = 'application/json'
        resp.set_headers(self.cors_headers)
        resp.status = falcon.HTTP_200
