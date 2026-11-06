"""Seekable HTTP file for reading zip central directories via range requests."""

import io
import urllib.request
import zipfile


class HttpFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.pos = 0
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "hasar-research"})
        with urllib.request.urlopen(req) as r:
            self.url = r.geturl()
            self.size = int(r.headers["Content-Length"])
        self.nbytes = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.pos}-{end}", "User-Agent": "hasar-research"}
        )
        with urllib.request.urlopen(req) as r:
            data = r.read()
        self.pos += len(data)
        self.nbytes += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[: len(d)] = d
        return len(d)


BASE = "https://huggingface.co/datasets/Kullervo/BRIGHT/resolve/main/"


def open_zip(name):
    f = HttpFile(BASE + name)
    return zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 20)), f
