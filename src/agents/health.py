import socket
import http.client

def is_ollama_running() -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    if sock.connect_ex(('localhost', 11434)) == 0:
        sock.close()
        conn = http.client.HTTPConnection("localhost", 11434, timeout=3)
        conn.request("GET", "/")
        res: http.client.HTTPResponse = conn.getresponse()
        data: bytes = res.read()
        conn.close()
        return res.status == 200 and b"Ollama is running" in data
    sock.close()
    return False