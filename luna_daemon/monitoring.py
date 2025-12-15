"""
Luna Daemon WebSocket Monitoring Server

Provides real-time status updates to the Luna JS panel in ComfyUI.
Compatible with LUNA-Narrates monitoring pattern.

Message Types:
- {"type": "status", "data": {...}}      - Periodic status updates
- {"type": "scaling", "data": {...}}     - Worker scale up/down events  
- {"type": "request", "data": {...}}     - Request started/completed
- {"type": "error", "data": {...}}       - Error events
"""

import socket
import struct
import threading
import hashlib
import base64
import json
import time
import os
from typing import Optional, Set, Tuple, Dict, Any, Callable

# Try relative import first, fallback to direct
try:
    from .core import logger
except (ImportError, ValueError):
    # Fallback: load core.py directly
    import importlib.util
    core_path = os.path.join(os.path.dirname(__file__), "core.py")
    spec = importlib.util.spec_from_file_location("luna_daemon_core", core_path)
    if spec and spec.loader:
        core_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core_mod)
        logger = core_mod.logger
    else:
        import logging
        logger = logging.getLogger("LunaMonitoring")


class WebSocketServer:
    """
    Simple WebSocket server for daemon status monitoring.
    
    Broadcasts status updates to connected clients (Luna JS panel).
    """
    
    def __init__(
        self, 
        status_provider: Callable[[], Dict[str, Any]],
        host: str = "127.0.0.1", 
        port: int = 19284
    ):
        """
        Initialize the WebSocket server.
        
        Args:
            status_provider: Callable that returns current daemon status dict
            host: Host to bind to
            port: Port to listen on
        """
        self.status_provider = status_provider
        self.host = host
        self.port = port
        self.clients: Set[socket.socket] = set()
        self.clients_lock = threading.Lock()
        self._running = False
        self._server_socket: Optional[socket.socket] = None
        self._broadcast_thread: Optional[threading.Thread] = None
        self._accept_thread: Optional[threading.Thread] = None
    
    def _create_accept_key(self, key: str) -> str:
        """Create WebSocket accept key from client key."""
        GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
        sha1 = hashlib.sha1((key + GUID).encode()).digest()
        return base64.b64encode(sha1).decode()
    
    def _handshake(self, conn: socket.socket) -> bool:
        """Perform WebSocket handshake."""
        try:
            data = conn.recv(4096).decode('utf-8')
            if not data:
                return False
            
            # Parse headers
            headers = {}
            lines = data.split('\r\n')
            for line in lines[1:]:
                if ':' in line:
                    key, value = line.split(':', 1)
                    headers[key.strip().lower()] = value.strip()
            
            # Check for WebSocket upgrade
            if headers.get('upgrade', '').lower() != 'websocket':
                return False
            
            # Get client key
            client_key = headers.get('sec-websocket-key', '')
            if not client_key:
                return False
            
            # Send handshake response
            accept_key = self._create_accept_key(client_key)
            response = (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept_key}\r\n"
                "\r\n"
            )
            conn.sendall(response.encode())
            return True
            
        except Exception as e:
            logger.error(f"WebSocket handshake error: {e}")
            return False
    
    def _encode_frame(self, data: str) -> bytes:
        """Encode data as WebSocket text frame."""
        payload = data.encode('utf-8')
        length = len(payload)
        
        if length <= 125:
            frame = bytes([0x81, length]) + payload
        elif length <= 65535:
            frame = bytes([0x81, 126]) + struct.pack('>H', length) + payload
        else:
            frame = bytes([0x81, 127]) + struct.pack('>Q', length) + payload
        
        return frame
    
    def _decode_frame(self, conn: socket.socket) -> Optional[str]:
        """Decode incoming WebSocket frame."""
        try:
            header = conn.recv(2)
            if len(header) < 2:
                return None
            
            opcode = header[0] & 0x0F
            
            # Close frame
            if opcode == 0x08:
                return None
            
            # Ping - send pong
            if opcode == 0x09:
                conn.sendall(bytes([0x8A, 0]))
                return ""
            
            masked = (header[1] & 0x80) != 0
            length = header[1] & 0x7F
            
            if length == 126:
                length = struct.unpack('>H', conn.recv(2))[0]
            elif length == 127:
                length = struct.unpack('>Q', conn.recv(8))[0]
            
            if masked:
                mask = conn.recv(4)
                data = bytearray(conn.recv(length))
                for i in range(length):
                    data[i] ^= mask[i % 4]
                return data.decode('utf-8')
            else:
                return conn.recv(length).decode('utf-8')
                
        except Exception:
            return None
    
    def broadcast(self, message_type: str, data: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        message = json.dumps({"type": message_type, "data": data})
        frame = self._encode_frame(message)
        
        with self.clients_lock:
            dead_clients = []
            for client in self.clients:
                try:
                    client.sendall(frame)
                except Exception:
                    dead_clients.append(client)
            
            # Clean up dead connections
            for client in dead_clients:
                self.clients.discard(client)
                try:
                    client.close()
                except:
                    pass
    
    def broadcast_event(self, event_type: str, **event_data):
        """Convenience method to broadcast a typed event."""
        self.broadcast(event_type, event_data)
    
    def _handle_client(self, conn: socket.socket, addr: Tuple[str, int]):
        """Handle a single WebSocket client connection."""
        if not self._handshake(conn):
            conn.close()
            return
        
        with self.clients_lock:
            self.clients.add(conn)
        
        logger.info(f"WebSocket client connected: {addr}")
        
        # Send initial status
        try:
            status = self.status_provider()
            message = json.dumps({"type": "status", "data": status})
            conn.sendall(self._encode_frame(message))
        except Exception as e:
            logger.error(f"Error sending initial status: {e}")
        
        # Keep connection alive and handle incoming messages
        try:
            while self._running:
                conn.settimeout(1.0)
                try:
                    data = self._decode_frame(conn)
                    if data is None:  # Connection closed
                        break
                    
                    # Handle client messages
                    if data:
                        try:
                            msg = json.loads(data)
                            if msg.get("type") == "get_status":
                                status = self.status_provider()
                                response = json.dumps({"type": "status", "data": status})
                                conn.sendall(self._encode_frame(response))
                        except json.JSONDecodeError:
                            pass
                            
                except socket.timeout:
                    continue
                except Exception:
                    break
                    
        finally:
            with self.clients_lock:
                self.clients.discard(conn)
            try:
                conn.close()
            except:
                pass
            logger.info(f"WebSocket client disconnected: {addr}")
    
    def _broadcast_loop(self):
        """Periodically broadcast status to all clients."""
        while self._running:
            time.sleep(1.0)  # Broadcast every second
            
            if self.clients:
                try:
                    status = self.status_provider()
                    self.broadcast("status", status)
                except Exception as e:
                    logger.error(f"Broadcast error: {e}")
    
    def _accept_loop(self):
        """Accept incoming connections."""
        while self._running:
            try:
                if self._server_socket is None:
                    break
                conn, addr = self._server_socket.accept()
                thread = threading.Thread(
                    target=self._handle_client,
                    args=(conn, addr),
                    daemon=True
                )
                thread.start()
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"WebSocket accept error: {e}")
    
    def start(self):
        """Start the WebSocket server."""
        self._running = True
        
        # Create server socket
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(5)
        self._server_socket.settimeout(1.0)
        
        # Start broadcast thread
        self._broadcast_thread = threading.Thread(
            target=self._broadcast_loop,
            name="ws-broadcast",
            daemon=True
        )
        self._broadcast_thread.start()
        
        # Start accept thread
        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            name="ws-accept",
            daemon=True
        )
        self._accept_thread.start()
        
        logger.info(f"WebSocket monitoring server started on ws://{self.host}:{self.port}")
    
    def stop(self):
        """Stop the WebSocket server."""
        self._running = False
        
        with self.clients_lock:
            for client in self.clients:
                try:
                    client.close()
                except:
                    pass
            self.clients.clear()
        
        if self._server_socket:
            try:
                self._server_socket.close()
            except:
                pass
            self._server_socket = None
        
        logger.info("WebSocket monitoring server stopped")
    
    @property
    def client_count(self) -> int:
        """Return number of connected clients."""
        with self.clients_lock:
            return len(self.clients)
    
    @property
    def is_running(self) -> bool:
        """Return whether server is running."""
        return self._running
