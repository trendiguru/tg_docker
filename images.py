import falcon
from bson import json_util
from rq import Queue

from tg_falcon import page_results
from trendi.constants import redis_conn, q1,

q1 = Queue('new_images', connection=redis_conn)


class Resource(object):
    cors_headers = {'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Credentials': 'true',
                    'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE',
                    'Access-Control-Max-Age': 86400}

    def on_get(self, req, resp):
        page_url = req.get_param("pageUrl")
        image_url = req.get_param("imageUrl")
        exists = req.get_param("exists")
        relevant = req.get_param("relevant")
        lang = req.get_param("lang")

        if lang:
            page_results.set_lang(lang)

        if page_url:
            # ret = page_results.get_all_data_for_page(page_url) or {"page_url": page_url}
            # ret = paperdolls.search_existing_images(page_url)
        elif image_url:
            if exists:
                ret = {"exists": page_results.image_exists(image_url)}
            elif relevant:
                ret = {"relevant": page_results.is_image_relevant(image_url)}
            else:
                # TODO handle case where image url has parameters that may change, e.g.:
                # http://image.gala.de/v1/cms/M-/familie-jolie-pitt-19jul_9085733-ORIGINAL-imageGallery_standard.jpg?v=11846425
                ret = page_results.get_data_for_specific_image(image_url=image_url)
        else:
            raise falcon.HTTPBadRequest("Try harder", "I need some params") # {"error": "Houston.."}

        resp.data = json_util.dumps(ret)
        resp.content_type = 'application/json'
        resp.status = falcon.HTTP_200


    def on_post(self, req, resp):
        ret = {"success": False}
        method = req.get_param('method') or "pd"
        try:
            data = json_util.loads(req.stream.read())
            page_url = data.get("pageUrl")
            images = data.get("imageList")
            lang = data.get("lang")
            coll_name = page_results.set_lang(lang)
            if type(images) is list and page_url is not None:
                relevancy_dict = {image_url:page_results.is_image_relevant(image_url, coll_name) for image_url in images}
                relevant_urls = [image_url for image_url in images if page_results.is_image_relevant(image_url, coll_name)]
                if method == 'qc':
                    [q1.enqueue(qcs.from_image_url_to_categorization_task, image_url) for image_url in images]
                elif method == 'pd':
                    [q1.enqueue_call(func=paperdolls.start_process, args=(page_url, image_url, lang), ttl=1000, result_ttl=1000, timeout=1000) for image_url in images]
                ret["success"] = True
                ret["relevant_urls"] = relevant_urls
                ret["relevancy_dict"] = relevancy_dict
            else:
                ret["success"] = False
                ret["error"] = "Missing image list and/or page url"

        except Exception as e:
            ret["error"] = str(e)
        
        resp.data = json_util.dumps(ret)
        resp.content_type = 'application/json'
        resp.status = falcon.HTTP_200
