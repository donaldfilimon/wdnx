import socket
import threading
import time
import logging
import struct


class NodeDiscovery:
    """
    Auto-discover WDBX nodes on the local network using multicast beacons.

    Usage:
        discovery = NodeDiscovery(service_port=8000, discovery_port=9999)
        discovery.start()
        peers = discovery.get_peers()
        discovery.stop()
    """
    MULTICAST_GROUP = '224.0.0.1'
    BROADCAST_INTERVAL = 5  # seconds

    def __init__(self, service_port: int, discovery_port: int = 9999):
        self.service_port = service_port
        self.discovery_port = discovery_port
        self._stop_event = threading.Event()
        self._peers = set()

    def start(self):
        """Start broadcasting and listening for peer beacons."""
        self._stop_event.clear()
        threading.Thread(target=self._broadcast_loop, daemon=True).start()
        threading.Thread(target=self._listen_loop, daemon=True).start()

    def stop(self):
        """Stop discovery threads."""
        self._stop_event.set()

    def _broadcast_loop(self):
        """Periodically broadcast this node's presence."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ttl = struct.pack('b', 1)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
        message = f'WDBX_PEER:{self.service_port}'.encode('utf-8')
        while not self._stop_event.is_set():
            try:
                sock.sendto(message, (self.MULTICAST_GROUP, self.discovery_port))
            except Exception as e:
                logging.error(f"Discovery broadcast error: {e}")
            time.sleep(self.BROADCAST_INTERVAL)

    def _listen_loop(self):
        """Listen for incoming peer beacons."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('', self.discovery_port))
        group = socket.inet_aton(self.MULTICAST_GROUP)
        mreq = group + socket.inet_aton('0.0.0.0')
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        while not self._stop_event.is_set():
            try:
                data, addr = sock.recvfrom(1024)
                if data.startswith(b'WDBX_PEER:'):
                    port = int(data.split(b':', 1)[1])
                    peer = (addr[0], port)
                    self._peers.add(peer)
            except Exception as e:
                logging.error(f"Discovery listen error: {e}")
                break

    def get_peers(self):
        """Return a list of discovered (ip, port) tuples."""
        return list(self._peers) 