import tornado
from tornado import gen, httpclient


class BaseHandler(tornado.web.RequestHandler):
    def __init__(self, application, request, **kwargs):
        super(BaseHandler, self).__init__(application, request, **kwargs)
        self.set_header('Content-Type', 'application/json; charset=UTF-8')
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Credentials", "true")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")

    @gen.coroutine
    def async_fetch(self, url, **kwargs):
        http_client = httpclient.AsyncHTTPClient()
        request = tornado.httpclient.HTTPRequest(url=url, connect_timeout=40.0, request_timeout=40.0, allow_nonstandard_methods=True,
                                                 headers={'Content-Type': 'application/json'}, **kwargs)

        res = yield gen.Task(http_client.fetch, request)
        self.write_async_log(res)
        return res


