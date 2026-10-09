"""Post one image to a running certo-vision server.

    python tools/post_image.py http://127.0.0.1:8080 demo/banana/samples/ripe.png ripeness overripe
"""
import base64
import json
import sys
import urllib.request

url, path, *qids = sys.argv[1:]
body = {"state": {"image_b64": base64.b64encode(open(path, "rb").read()).decode()}, "questions": qids}
req = urllib.request.Request(url.rstrip("/") + "/v1/systemone", json.dumps(body).encode(), {"Content-Type": "application/json"})
print(json.dumps(json.load(urllib.request.urlopen(req)), indent=1))
