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
        # page_url = req.get_param("pageUrl")
        # image_url = req.get_param("imageUrl")
        # exists = req.get_param("exists")
        # relevant = req.get_param("relevant")
        # lang = req.get_param("lang")
        #
        # if lang:
        #     page_results.set_lang(lang)
        #
        # if 'fashionseoul' in image_url:
        #     products = "GangnhamStyle"
        # else:
        #     products = "ShopStyle"
        # # products = determine with pageUrl
        #
        # ret = page_results.get_data_for_specific_image(image_url=image_url, products_collection=products)

        ret = {"hello": "world!", 
                "collection_names": constants.db.collection_names()}

        resp.data = json_util.dumps(ret) + "\n"
        resp.content_type = 'application/json'
        resp.set_headers(cors_headers)
        resp.status = falcon.HTTP_200

    def on_post(self, req, resp):
        ret = {"success": False}
        try:
            data = json_util.loads(req.stream.read())
            page_url = data.get("pageUrl")
            images = data.get("imageList")
            lang = data.get("lang")
            coll_name = page_results.set_lang(lang)
            if type(images) is list and page_url is not None:
                relevancy_dict = {}
                new_images = []
                for url in images:
                    if page_results.is_image_relevant(url, coll_name):
                        relevancy_dict[url] = True
                    else:
                        new_images.append(url)
                        relevancy_dict[url] = False
                    [q1.enqueue_call(func='fiction.function', args=(page_url, image_url, lang), ttl=2000,
                                     result_ttl=2000, timeout=2000) for image_url in new_images]
                ret["success"] = True
                ret["relevancy_dict"] = relevancy_dict
            else:
                ret["success"] = False
                ret["error"] = "Missing image list and/or page url"

        except Exception as e:
            ret["error"] = str(e)
        
        resp.data = json_util.dumps(ret)
        resp.content_type = 'application/json'
        resp.set_headers(cors_headers)
        resp.status = falcon.HTTP_200
