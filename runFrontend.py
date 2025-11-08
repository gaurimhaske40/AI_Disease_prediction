from http.server import SimpleHTTPRequestHandler, HTTPServer
import os
import threading
import time
import ssl
# Define server address and port
HOST = "localhost"
PORT = 7891

# Set up HTTP server
def run_server(HOST,PORT):
    os.chdir("webBuild")
    server_address = (HOST, PORT)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)

    link = f"http://{HOST}:{PORT}"
    print(f"Link : {link}")
    def openLink():
        time.sleep(2.0)
        os.system(f"start chrome -incognito {link}")
        pass
    threading.Thread(target=openLink).start()
    httpd.serve_forever()

if __name__ == "__main__":
    run_server(HOST=HOST,PORT=PORT)

