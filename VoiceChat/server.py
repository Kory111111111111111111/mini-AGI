#!/usr/bin/env python3
"""
Sesame Voice Arena - Lightweight Local Server
Serves the VoiceChat web application with zero external dependencies.
"""

import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 3000

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS and caching headers for smooth Web Audio testing
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

def run_server():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    global PORT
    
    # Try port 3000, or increment if busy
    for attempt in range(5):
        try:
            with socketserver.TCPServer(("", PORT), Handler) as httpd:
                print(f"\n" + "="*60)
                print(f"🎙️  Sesame Voice Arena is live!")
                print(f"👉 Local URL: http://localhost:{PORT}")
                print(f"="*60 + "\n")
                
                # Auto-open browser
                webbrowser.open(f"http://localhost:{PORT}")
                httpd.serve_forever()
                break
        except OSError:
            PORT += 1

if __name__ == "__main__":
    try:
        run_server()
    except KeyboardInterrupt:
        print("\nShutting down Sesame Voice Arena server.")
        sys.exit(0)
