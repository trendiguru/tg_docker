import os
import pymongo
from redis import Redis
from rq import Queue

redis_conn = Redis(host=os.getenv("REDIS_HOST", "redis1-redis-1-vm"), port=int(os.getenv("REDIS_PORT", "6379")))

manual_gender_domains = ['fashionseoul.com', 'haaretz.co.il']

products_per_site = {'default': 'ShopStyle', 'fashionseoul.com': 'GangnamStyle', 'fazz.co': 'ShopStyle'}

products_per_country = {'default': 'ebay', 'ebay': ['US'], 'GangnhamStyle': ['KR']}

q1 = Queue('start_pipeline', connection=redis_conn)

db = pymongo.MongoClient(host=os.getenv("MONGO_HOST", "mongodb"),
                         port=int(os.getenv("MONGO_PORT", "27017"))).mydb
