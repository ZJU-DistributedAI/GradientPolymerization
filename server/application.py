import sys
import tornado
from tornado.web import Application
import urls

class Application(tornado.web.Application):
    def __init__(self):
        handler = urls.urlpattern
        super(Application, self).__init__(handler)

def main(port):
    app = Application()
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(port)
    tornado.ioloop.IOLoop.instance().start()



if __name__ == "__main__":
    try:
        port = sys.argv[1]
    except IndexError:
        port = 8888
    main(port)