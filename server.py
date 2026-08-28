"""
server.py - Local server for lyrics site

Serves static files and song processing API.

Usage:
  .venv-whisperx/Scripts/python.exe server.py
"""

import http.server
import json
import os
import sys
import threading
import traceback

PORT = 8000
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

# ? tools/ ?? path ?? import
sys.path.insert(0, os.path.join(ROOT, 'tools'))


class SongHandler(http.server.SimpleHTTPRequestHandler):

    def do_POST(self):
        if self.path == '/api/add-song':
            self._handle_add_song()
        elif self.path == '/api/import-zh':
            self._handle_import_zh()
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path == '/api/status':
            self._json_response({'status': 'ready'})
        else:
            super().do_GET()

    def _read_body(self):
        length = int(self.headers.get('Content-Length', 0))
        return json.loads(self.rfile.read(length).decode('utf-8'))

    def _json_response(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def _handle_add_song(self):
        try:
            data = self._read_body()
            import add_song_new as asn

            lyrics = data.get('lyrics', [])
            zh = data.get('zh', [])

            song = {
                "title": data.get("title", ""),
                "artist": data.get("artist", ""),
                "credits": {
                    "lyrics": data.get("credits", {}).get("lyrics"),
                    "music": data.get("credits", {}).get("music"),
                },
                "color": data.get("color", "#c9a96e"),
                "videoId": data.get("videoId", ""),
                "lines": [{"time": 0, "text": l} for l in lyrics],
            }

            for i, z in enumerate(zh):
                if i < len(song["lines"]) and z:
                    song["lines"][i]["zh"] = z

            song = asn.process_song(song, do_translate=True, device="cpu")

            songs_path = os.path.join(ROOT, 'songs.json')
            if os.path.exists(songs_path):
                with open(songs_path, 'r', encoding='utf-8') as f:
                    songs = json.load(f)
            else:
                songs = []

            songs.append(song)
            with open(songs_path, 'w', encoding='utf-8') as f:
                json.dump(songs, f, ensure_ascii=False, indent=2)

            self._json_response({
                'success': True,
                'message': f"'{song['title']}' added! Total {len(songs)} songs, {len(song['lines'])} lines.",
                'lines': len(song['lines']),
            })

        except Exception as e:
            traceback.print_exc()
            self._json_response({'success': False, 'error': str(e)}, 500)

    def _handle_import_zh(self):
        try:
            data = self._read_body()
            title = data.get('title', '')
            translations = data.get('translations', [])

            songs_path = os.path.join(ROOT, 'songs.json')
            with open(songs_path, 'r', encoding='utf-8') as f:
                songs = json.load(f)

            found = False
            for song in songs:
                if song.get('title') == title:
                    found = True
                    for i, zh in enumerate(translations):
                        if i < len(song['lines']):
                            song['lines'][i]['zh'] = zh
                    break

            if not found:
                self._json_response({'success': False, 'error': f"Song not found: {title}"}, 404)
                return

            with open(songs_path, 'w', encoding='utf-8') as f:
                json.dump(songs, f, ensure_ascii=False, indent=2)

            self._json_response({
                'success': True,
                'message': f"Imported {len(translations)} lines to '{title}'",
            })

        except Exception as e:
            traceback.print_exc()
            self._json_response({'success': False, 'error': str(e)}, 500)


def main():
    server = http.server.ThreadingHTTPServer(('', PORT), SongHandler)
    print(f"Server started at http://localhost:{PORT}/")
    print(f"Add song page: http://localhost:{PORT}/add_song.html")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
        server.server_close()


if __name__ == '__main__':
    main()
