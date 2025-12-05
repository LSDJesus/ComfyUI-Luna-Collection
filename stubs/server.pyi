"""Type stubs for ComfyUI server module."""

from typing import Any
from aiohttp import web

class PromptServer:
    instance: 'PromptServer'
    app: web.Application
    routes: web.RouteTableDef
    
    def send_sync(self, event: str, data: Any, sid: str = ...) -> None: ...
