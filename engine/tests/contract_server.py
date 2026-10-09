"""Serve the real HTTP API with model stand-ins, for the CLI's contract test.

Usage: python tests/contract_server.py PORT DATA_DIRECTORY
"""

import sys

import uvicorn
from fakes import FakeRecognizer, IdleEngine

from idiolect_engine.server import create_app

port, data = int(sys.argv[1]), sys.argv[2]
server = None


def request_shutdown():
    server.should_exit = True


app = create_app(data, IdleEngine, FakeRecognizer(), request_shutdown)
server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
server.run()
