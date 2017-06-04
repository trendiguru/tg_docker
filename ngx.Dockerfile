#NEVER MIND with this docker file, thee following seems to wokrk

docker run --name some-nginx -v /data/www:/usr/share/nginx/html:ro -p 8090:80 -d  nginx

#ttps://blog.docker.com/2015/04/tips-for-deploying-nginx-official-image-with-docker/
FROM ubuntu:14.04
RUN apt-get update && apt-get install -y nginx
CMD ["/usr/sbin/nginx"]
CMD ["nginx", "-g", "daemon off;"]