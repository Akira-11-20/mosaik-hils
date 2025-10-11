"""
カスタムWeb可視化サーバー
Mosaik HILSシミュレーション用の独自Webダッシュボードサーバー
"""

import json
import asyncio
import websockets
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
from pathlib import Path
import mosaik


class CustomWebServer:
    """カスタムWeb可視化サーバー"""
    
    def __init__(self, host='localhost', http_port=8003, ws_port=8004):
        self.host = host
        self.http_port = http_port
        self.ws_port = ws_port
        self.clients = set()
        self.simulation_data = {
            'numerical': 0,
            'sensor': 0,
            'actuator': 0,
            'timestamp': 0
        }
        
    def start_servers(self):
        """HTTPサーバーとWebSocketサーバーを開始"""
        # HTTPサーバーを別スレッドで開始
        http_thread = threading.Thread(target=self._start_http_server)
        http_thread.daemon = True
        http_thread.start()
        
        # WebSocketサーバーを開始
        asyncio.run(self._start_websocket_server())
    
    def _start_http_server(self):
        """静的ファイルを提供するHTTPサーバー"""
        os.chdir(Path(__file__).parent / 'web_visualization')
        
        class CustomHandler(SimpleHTTPRequestHandler):
            def end_headers(self):
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.send_header('Pragma', 'no-cache')
                self.send_header('Expires', '0')
                super().end_headers()
        
        server = HTTPServer((self.host, self.http_port), CustomHandler)
        print(f"📡 カスタムHTTPサーバー開始: http://{self.host}:{self.http_port}")
        server.serve_forever()
    
    async def _start_websocket_server(self):
        """WebSocketサーバー"""
        print(f"🔌 WebSocketサーバー開始: ws://{self.host}:{self.ws_port}")
        await websockets.serve(self._websocket_handler, self.host, self.ws_port)
        await asyncio.Future()  # 永続実行
    
    async def _websocket_handler(self, websocket, path):
        """WebSocket接続ハンドラー"""
        self.clients.add(websocket)
        print(f"✅ クライアント接続: {websocket.remote_address}")
        
        try:
            # 接続時に現在のデータを送信
            await websocket.send(json.dumps({
                'type': 'initial_data',
                'data': self.simulation_data
            }))
            
            async for message in websocket:
                # クライアントからのメッセージ処理
                try:
                    data = json.loads(message)
                    await self._handle_client_message(data, websocket)
                except json.JSONDecodeError:
                    print(f"❌ 無効なJSONメッセージ: {message}")
        
        except websockets.exceptions.ConnectionClosed:
            print(f"📴 クライアント切断: {websocket.remote_address}")
        finally:
            self.clients.remove(websocket)
    
    async def _handle_client_message(self, data, websocket):
        """クライアントメッセージの処理"""
        message_type = data.get('type')
        
        if message_type == 'command':
            command = data.get('command')
            params = data.get('params', {})
            print(f"📥 コマンド受信: {command} {params}")
            
            # コマンド処理（実装例）
            if command == 'start':
                await self._broadcast({'type': 'status', 'message': 'シミュレーション開始'})
            elif command == 'pause':
                await self._broadcast({'type': 'status', 'message': 'シミュレーション一時停止'})
            elif command == 'speed':
                speed_factor = params.get('factor', 1.0)
                await self._broadcast({
                    'type': 'status', 
                    'message': f'速度変更: {speed_factor}x'
                })
    
    async def _broadcast(self, message):
        """全クライアントにメッセージをブロードキャスト"""
        if self.clients:
            disconnected = set()
            for client in self.clients:
                try:
                    await client.send(json.dumps(message))
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(client)
            
            # 切断されたクライアントを削除
            self.clients -= disconnected
    
    def update_data(self, data):
        """シミュレーションデータの更新"""
        self.simulation_data.update(data)
        
        # 非同期でクライアントに送信
        if self.clients:
            asyncio.create_task(self._broadcast({
                'type': 'data_update',
                'data': self.simulation_data
            }))


class CustomWebVisSimulator(mosaik.Simulator):
    """Mosaik用カスタムWeb可視化シミュレーター"""
    
    def __init__(self):
        super().__init__({'WebVisualization': {'WebDashboard': []}})
        self.web_server = None
        self.dashboard = None
    
    def init(self, sid, host='localhost', http_port=8003, ws_port=8004):
        """シミュレーター初期化"""
        self.web_server = CustomWebServer(host, http_port, ws_port)
        
        # サーバーを別スレッドで開始
        server_thread = threading.Thread(target=self.web_server.start_servers)
        server_thread.daemon = True
        server_thread.start()
        
        return {'WebVisualization': 'WebDashboard'}
    
    def create(self, num, model, **model_conf):
        """エンティティ作成"""
        entities = []
        for i in range(num):
            eid = f'WebDashboard_{i}'
            entities.append({'eid': eid, 'type': model})
        
        self.dashboard = entities[0]['eid'] if entities else None
        return entities
    
    def step(self, time, inputs):
        """シミュレーションステップ実行"""
        if self.dashboard and self.dashboard in inputs:
            # 入力データを取得
            data = inputs[self.dashboard]
            
            # データを整理してWebサーバーに送信
            processed_data = {
                'timestamp': time
            }
            
            for attr, values in data.items():
                if values:  # 値が存在する場合
                    # 複数の値がある場合は最新値を使用
                    processed_data[attr] = list(values.values())[0]
            
            # Webサーバーにデータを送信
            if self.web_server:
                self.web_server.update_data(processed_data)
        
        return time + 1  # 次のステップ時間を返す
    
    def get_data(self, outputs):
        """データ取得（必要に応じて実装）"""
        return {}


def main():
    """独立実行用のメイン関数"""
    print("🚀 カスタムWeb可視化サーバー単体起動")
    server = CustomWebServer()
    server.start_servers()


if __name__ == '__main__':
    main()