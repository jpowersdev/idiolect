"""Serve the real HTTP API with model stand-ins, for the CLI's contract test.

Usage: python tests/contract_server.py DATA_DIRECTORY
"""

import socket
import sys

import uvicorn
from fakes import FakeRecognizer, IdleEngine

from idiolect_engine.config import socket_path
from idiolect_engine.server import create_app

data = sys.argv[1]
server = None


def request_shutdown():
    server.should_exit = True


listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
listener.bind(str(socket_path(data)))
app = create_app(data, IdleEngine, FakeRecognizer(), request_shutdown)
server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
server.run(sockets=[listener])
