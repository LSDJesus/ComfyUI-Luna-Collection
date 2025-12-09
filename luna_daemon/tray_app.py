"""
Luna Daemon - System Tray Application
Single-instance daemon that runs in Windows system tray.

Features:
- Single instance enforcement (only one daemon per machine)
- System tray icon with status indicator
- Right-click menu: Start/Stop/Status/View Logs/Exit
- Auto-start with Windows (optional)
- Persistent across ComfyUI sessions

Requirements: pip install pystray pillow
"""

import os
import sys
import threading
import time
import socket
import logging
from pathlib import Path
from datetime import datetime

try:
    import pystray
    from PIL import Image, ImageDraw
    HAS_TRAY = True
except ImportError:
    HAS_TRAY = False
    print("Warning: pystray not installed. Run: pip install pystray pillow")

# Set up file logging (when running hidden in background)
LOG_FILE = Path(__file__).parent / "tray_app.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)

# Single-instance lock using a socket
LOCK_PORT = 19299  # Different from daemon ports
LOCK_HOST = '127.0.0.1'

class SingleInstanceLock:
    """Ensures only one instance of the daemon tray app runs"""
    
    def __init__(self):
        self.sock = None
    
    def acquire(self) -> bool:
        """Try to acquire the lock. Returns True if successful."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
            self.sock.bind((LOCK_HOST, LOCK_PORT))
            self.sock.listen(1)
            return True
        except OSError:
            # Port already in use - another instance is running
            self.sock = None
            return False
    
    def release(self):
        """Release the lock"""
        if self.sock:
            self.sock.close()
            self.sock = None


class LunaDaemonTray:
    """System tray application for Luna Daemon"""
    
    def __init__(self):
        self.lock = SingleInstanceLock()
        self.daemon_thread = None
        self.daemon_running = False
        self.icon = None
        
        # Import daemon server
        daemon_dir = Path(__file__).parent
        if str(daemon_dir) not in sys.path:
            sys.path.insert(0, str(daemon_dir))
        
        try:
            from server import DynamicDaemon
            from config import SHARED_DEVICE, CLIP_PRECISION, VAE_PRECISION, ServiceType
            self.DynamicDaemon = DynamicDaemon
            self.SHARED_DEVICE = SHARED_DEVICE
            self.CLIP_PRECISION = CLIP_PRECISION
            self.VAE_PRECISION = VAE_PRECISION
            self.ServiceType = ServiceType
        except ImportError as e:
            print(f"Error importing daemon: {e}")
            self.DynamicDaemon = None
    
    def create_icon_image(self, color='green'):
        """Create a simple colored circle icon"""
        size = 64
        image = Image.new('RGB', (size, size), color='white')
        dc = ImageDraw.Draw(image)
        
        # Draw circle
        color_map = {
            'green': '#00ff00',
            'red': '#ff0000',
            'yellow': '#ffff00',
            'gray': '#808080'
        }
        dc.ellipse([4, 4, size-4, size-4], fill=color_map.get(color, '#808080'))
        
        return image
    
    def check_daemon_status(self) -> bool:
        """Check if daemon is actually responding"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            result = sock.connect_ex(('127.0.0.1', 19283))
            sock.close()
            return result == 0
        except:
            return False
    
    def start_daemon(self, icon=None, item=None):
        """Start the daemon in a background thread"""
        if self.daemon_running:
            return
        
        if not self.DynamicDaemon:
            print("Daemon not available")
            return
        
        def run_daemon():
            try:
                print("Starting Luna Daemon...")
                self.daemon_running = True
                if self.icon:
                    self.icon.icon = self.create_icon_image('yellow')
                    self.icon.title = "Luna Daemon - Starting..."
                
                daemon = self.DynamicDaemon(
                    device=self.SHARED_DEVICE,
                    clip_precision=self.CLIP_PRECISION,
                    vae_precision=self.VAE_PRECISION,
                    service_type=self.ServiceType.FULL
                )
                
                if self.icon:
                    self.icon.icon = self.create_icon_image('green')
                    self.icon.title = "Luna Daemon - Running"
                
                daemon.run()
            except Exception as e:
                print(f"Daemon error: {e}")
                self.daemon_running = False
                if self.icon:
                    self.icon.icon = self.create_icon_image('red')
                    self.icon.title = f"Luna Daemon - Error: {str(e)[:50]}"
        
        self.daemon_thread = threading.Thread(target=run_daemon, daemon=True)
        self.daemon_thread.start()
    
    def stop_daemon(self, icon=None, item=None):
        """Stop the daemon cleanly"""
        logger.info("Stopping daemon...")
        try:
            # Use socket directly to send shutdown command
            # This avoids all import issues
            import socket
            import json
            
            daemon_host = "127.0.0.1"
            daemon_port = 19283
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)  # Shorter timeout for shutdown
            
            try:
                sock.connect((daemon_host, daemon_port))
            except (socket.timeout, ConnectionRefusedError):
                logger.warning("Daemon not running or not responding")
                self.daemon_running = False
                if self.icon:
                    self.icon.icon = self.create_icon_image('gray')
                    self.icon.title = "Luna Daemon - Stopped"
                    self.icon.notify("Daemon Stopped", "Already stopped")
                return
            
            # Send shutdown command
            cmd = json.dumps({"cmd": "shutdown"})
            sock.sendall((cmd + "\n").encode())
            
            # Don't wait for response - daemon closes connection after shutdown
            sock.close()
            
            self.daemon_running = False
            
            if self.icon:
                self.icon.icon = self.create_icon_image('gray')
                self.icon.title = "Luna Daemon - Stopped"
                self.icon.notify("Daemon Stopped", "Shutdown complete")
            
            logger.info("Daemon stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping daemon: {e}")
            if self.icon:
                # Keep error message short (max 64 chars for Windows notification)
                error_msg = str(e)[:35]
                self.icon.notify("Stop Failed", error_msg)
    
    def show_status(self, icon=None, item=None):
        """Show daemon status"""
        status = self.check_daemon_status()
        if status:
            msg = "✓ Daemon is running and responsive"
            if self.icon:
                self.icon.icon = self.create_icon_image('green')
        else:
            msg = "✗ Daemon is not responding"
            if self.icon:
                self.icon.icon = self.create_icon_image('red')
        
        print(msg)
        if self.icon:
            self.icon.notify("Luna Daemon Status", msg)
    
    def view_logs(self, icon=None, item=None):
        """Open log file"""
        import subprocess
        log_file = Path(__file__).parent / "tray_app.log"
        
        if not log_file.exists():
            if self.icon:
                self.icon.notify("Logs", "No logs yet")
            return
        
        try:
            # Open log file in default text editor
            subprocess.Popen(f'notepad "{log_file}"')
            logger.info(f"Opened logs: {log_file}")
        except Exception as e:
            logger.error(f"Could not open logs: {e}")
            if self.icon:
                self.icon.notify("Error", "Could not open logs")
    
    def quit_app(self, icon=None, item=None):
        """Quit the tray application"""
        print("Exiting Luna Daemon tray app...")
        self.daemon_running = False
        self.lock.release()
        if icon:
            icon.stop()
    
    def run(self):
        """Run the system tray application"""
        if not HAS_TRAY:
            print("Error: pystray not installed")
            print("Install with: pip install pystray pillow")
            return
        
        # Try to acquire single-instance lock
        if not self.lock.acquire():
            print("Luna Daemon is already running!")
            print("Check your system tray for the Luna icon.")
            return
        
        print("Luna Daemon Tray App starting...")
        
        # Create system tray icon
        icon_image = self.create_icon_image('gray')
        
        menu = pystray.Menu(
            pystray.MenuItem('Start Daemon', self.start_daemon),
            pystray.MenuItem('Stop Daemon', self.stop_daemon),
            pystray.MenuItem('Status', self.show_status),
            pystray.MenuItem('View Logs', self.view_logs),
            pystray.MenuItem('Exit', self.quit_app)
        )
        
        self.icon = pystray.Icon(
            "Luna Daemon",
            icon_image,
            "Luna Daemon - Stopped",
            menu
        )
        
        # Auto-start the daemon
        threading.Timer(1.0, self.start_daemon).start()
        
        # Run the icon (blocking)
        try:
            self.icon.run()
        except KeyboardInterrupt:
            self.quit_app()
        finally:
            self.lock.release()


def main():
    """Entry point"""
    app = LunaDaemonTray()
    app.run()


if __name__ == "__main__":
    main()
