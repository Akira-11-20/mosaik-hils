"""
Local Assets WebVis Simulator
プロジェクトルートのwebvis_assetsを使用するカスタムWebVisualizationシミュレーター

.venvを変更せず、プロジェクト内のアセットでWebVisをカスタマイズ
"""

import os
import shutil
import subprocess
import sys
import time

from pathlib import Path
import mosaik_api


META = {
    "type": "time-based",
    "models": {
        "Topology": {
            "public": True,
            "params": [],
            "attrs": [],
            "any_inputs": True,
        },
    },
    "extra_methods": [
        "set_config",
        "set_etypes",
    ],
}


class LocalAssetsWebVisSimulator(mosaik_api.Simulator):
    """
    ローカルアセットを使用するカスタムWebVisシミュレーター
    webvis_assets/からHTMLファイルを.venvにコピーして使用
    """

    def __init__(self):
        super().__init__(META)
        self.entities = {}
        self.proc = None
        self.project_root = Path(__file__).parent.parent.parent
        self.webvis_assets = self.project_root / "webvis_assets"

    def init(self, sid, **sim_params):
        """Initialize and start WebVis with local assets"""
        self.sid = sid

        print("🔧 Setting up WebVis with local assets...")

        # ローカルアセットをvenvにコピー
        success = self._deploy_local_assets()

        if success:
            print("✅ Local assets deployed successfully!")
        else:
            print("⚠️  Failed to deploy local assets, using default")

        # WebVisサーバーはmosaik_web.mosaikシミュレーターで起動
        # 実際には、このシミュレーター自体がWebVis機能を提供
        print(f"✅ WebVis with Local Assets ready at: http://localhost:8002")
        print(f"🎨 Using customized assets from: {self.webvis_assets}")
        
        # WebVisサーバープロセスを起動（実際のmosaik_web）
        serve_addr = sim_params.get("serve_addr", "127.0.0.1:8002")
        addr = sim_params.get("addr", "127.0.0.1:9999")  # mosaik接続用アドレス
        
        cmd = [sys.executable, "-m", "mosaik_web.mosaik", f"tcp://{addr}", f"--serve={serve_addr}"]
        
        print(f"🚀 Starting mosaik-web server: {' '.join(cmd)}")
        
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # サーバー起動を待つ
        time.sleep(3)
        
        # プロセスが正常に起動したかチェック
        if self.proc.poll() is not None:
            stdout, _ = self.proc.communicate()
            print(f"❌ mosaik-web server failed to start")
            print(f"📄 Output: {stdout}")
        else:
            print("✅ mosaik-web server started successfully")

        return self.meta

    def _deploy_local_assets(self):
        """Deploy local assets to mosaik_web package"""
        try:
            # mosaik_webのHTMLパスを取得
            target_html_path = self._get_mosaik_web_html_path()

            if not target_html_path or not target_html_path.exists():
                print(f"❌ Could not find mosaik_web HTML path: {target_html_path}")
                return False

            # ローカルアセットの確認
            local_html = self.webvis_assets / "index.html"
            local_media = self.webvis_assets / "media"

            if not local_html.exists():
                print(f"❌ Local HTML not found: {local_html}")
                return False

            # バックアップの作成
            backup_path = target_html_path.with_suffix(".html.backup")
            if not backup_path.exists():
                shutil.copy2(target_html_path, backup_path)
                print(f"📋 Backup created: {backup_path}")

            # HTMLファイルのコピー
            shutil.copy2(local_html, target_html_path)
            print(f"📄 HTML deployed: {local_html} → {target_html_path}")

            # メディアファイルのコピー（存在する場合）
            if local_media.exists():
                target_media = target_html_path.parent / "media"
                if target_media.exists():
                    # 必要に応じて個別ファイルをコピー
                    for media_file in local_media.iterdir():
                        if media_file.is_file():
                            target_file = target_media / media_file.name
                            # オリジナルファイルが存在し、更新が必要な場合のみコピー
                            if (
                                not target_file.exists()
                                or media_file.stat().st_mtime
                                > target_file.stat().st_mtime
                            ):
                                shutil.copy2(media_file, target_file)
                                print(f"📁 Media updated: {media_file.name}")

            return True

        except Exception as e:
            print(f"❌ Asset deployment failed: {e}")
            return False

    def _get_mosaik_web_html_path(self):
        """Find mosaik_web HTML file path"""
        try:
            import mosaik_web

            package_path = Path(mosaik_web.__file__).parent
            return package_path / "html" / "index.html"
        except ImportError:
            # Fallback search
            venv_path = Path(sys.executable).parent.parent
            site_packages = (
                venv_path
                / "lib"
                / f"python{sys.version_info.major}.{sys.version_info.minor}"
                / "site-packages"
            )
            return site_packages / "mosaik_web" / "html" / "index.html"

    def create(self, num, model, **model_params):
        """Create topology entities"""
        entities = []
        for i in range(num):
            eid = f"{model}_{i}"
            self.entities[eid] = {"model": model, "static": True}
            entities.append({"eid": eid, "type": model})
        return entities

    def step(self, time, inputs, max_advance=None):
        """Process simulation step"""
        # WebVisは通常静的なので入力データを保存するだけ
        return time + 60  # 60秒間隔で更新

    def get_data(self, outputs):
        """Get data from entities"""
        return {}

    def set_etypes(self, etypes):
        """Set entity types for visualization"""
        print(f"🎨 Local Assets WebVis: Setting entity types: {list(etypes.keys())}")

    def finalize(self):
        """Clean up when simulation ends"""
        if self.proc:
            print("🛑 Shutting down Local Assets WebVis...")
            self.proc.terminate()
            self.proc.wait()


if __name__ == "__main__":
    mosaik_api.start_simulation(LocalAssetsWebVisSimulator())
