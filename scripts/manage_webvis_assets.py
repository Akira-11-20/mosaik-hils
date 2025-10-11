#!/usr/bin/env python3
"""
WebVis Assets Management Script
プロジェクトルートのwebvis_assetsを管理するスクリプト

使用方法:
    python scripts/manage_webvis_assets.py --deploy     # ローカルアセットを.venvに展開
    python scripts/manage_webvis_assets.py --restore    # 元のアセットに復元
    python scripts/manage_webvis_assets.py --status     # 現在の状態確認
    python scripts/manage_webvis_assets.py --create     # 初期アセット作成
"""

import sys
import argparse
import shutil
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def get_paths():
    """Get relevant file paths"""
    try:
        import mosaik_web

        package_path = Path(mosaik_web.__file__).parent
        target_html = package_path / "html" / "index.html"
        target_media = package_path / "html" / "media"
    except ImportError:
        # Fallback search
        venv_path = Path(sys.executable).parent.parent
        site_packages = (
            venv_path
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
        target_html = site_packages / "mosaik_web" / "html" / "index.html"
        target_media = site_packages / "mosaik_web" / "html" / "media"

    local_assets = project_root / "webvis_assets"
    local_html = local_assets / "index.html"
    local_media = local_assets / "media"
    backup_html = target_html.with_suffix(".html.backup")

    return {
        "target_html": target_html,
        "target_media": target_media,
        "local_assets": local_assets,
        "local_html": local_html,
        "local_media": local_media,
        "backup_html": backup_html,
    }


def check_status():
    """Check current WebVis assets status"""
    print("🔍 WebVis Assets Status Check")
    print("-" * 50)

    paths = get_paths()

    # Local assets check
    if paths["local_assets"].exists():
        print(f"📁 Local Assets: {paths['local_assets']} ✅")
        if paths["local_html"].exists():
            print(f"📄 Local HTML: {paths['local_html']} ✅")
        else:
            print(f"📄 Local HTML: {paths['local_html']} ❌")

        if paths["local_media"].exists():
            media_files = list(paths["local_media"].iterdir())
            print(f"📂 Local Media: {len(media_files)} files ✅")
        else:
            print(f"📂 Local Media: {paths['local_media']} ❌")
    else:
        print(f"📁 Local Assets: {paths['local_assets']} ❌")

    # Target check
    if paths["target_html"].exists():
        print(f"🎯 Target HTML: {paths['target_html']} ✅")

        # Check if using local assets
        try:
            content = paths["target_html"].read_text(encoding="utf-8")
            if "Local Assets" in content:
                print("🔧 Status: Using Local Assets ✅")
            elif "HILS Custom" in content:
                print("🔧 Status: Using Direct Customization")
            else:
                print("🔧 Status: Using Original")
        except Exception as e:
            print(f"🔧 Status: Error reading - {e}")
    else:
        print(f"🎯 Target HTML: {paths['target_html']} ❌")

    # Backup check
    if paths["backup_html"].exists():
        print(f"💾 Backup: {paths['backup_html']} ✅")
    else:
        print(f"💾 Backup: {paths['backup_html']} ❌")

    print("-" * 50)


def create_initial_assets():
    """Create initial local assets from mosaik_web"""
    print("🏗️  Creating initial local assets...")

    paths = get_paths()

    if not paths["target_html"].exists():
        print(f"❌ mosaik_web HTML not found: {paths['target_html']}")
        return False

    # Create local assets directory
    paths["local_assets"].mkdir(exist_ok=True)

    # Copy HTML
    if paths["backup_html"].exists():
        # Use backup (original) if available
        shutil.copy2(paths["backup_html"], paths["local_html"])
        print(
            f"📄 HTML copied from backup: {paths['backup_html']} → {paths['local_html']}"
        )
    else:
        shutil.copy2(paths["target_html"], paths["local_html"])
        print(
            f"📄 HTML copied from target: {paths['target_html']} → {paths['local_html']}"
        )

    # Copy media directory
    if paths["target_media"].exists():
        if paths["local_media"].exists():
            shutil.rmtree(paths["local_media"])
        shutil.copytree(paths["target_media"], paths["local_media"])
        print(f"📂 Media copied: {paths['target_media']} → {paths['local_media']}")

    # Apply basic customization
    try:
        content = paths["local_html"].read_text(encoding="utf-8")
        content = content.replace(
            "<title>mosaik-web</title>",
            "<title>mosaik-web (HILS Custom - Local Assets)</title>",
        )

        # Add delay statistics panel
        delay_stats_html = """<div id="delay-stats" style="position: absolute; top: 10px; right: 10px; background: rgba(255,255,255,0.9); padding: 10px; border-radius: 5px; font-family: monospace; font-size: 12px; z-index: 1000; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border: 1px solid #ddd;">
    <strong>🔄 Delay Node Statistics</strong><br>
    <span id="delay-info">Initializing...</span><br>
    <div id="delay-metrics" style="margin-top: 5px; font-size: 11px;">
        <div>📊 Packets: <span id="packets-info">-</span></div>
        <div>⏱️ Avg Delay: <span id="avg-delay-info">-</span></div>
        <div>📈 Jitter: <span id="jitter-info">-</span></div>
        <div style="margin-top: 3px; font-size: 10px; color: #666;">
            📂 Assets: Local (Root)
        </div>
    </div>
</div>"""

        content = content.replace(
            '<div id="progress"></div>',
            f'<div id="progress"></div>\n{delay_stats_html}',
        )

        paths["local_html"].write_text(content, encoding="utf-8")
        print("🎨 Basic customization applied to local HTML")

    except Exception as e:
        print(f"⚠️  Customization failed: {e}")

    print("✅ Initial local assets created successfully!")
    return True


def deploy_assets():
    """Deploy local assets to mosaik_web"""
    print("🚀 Deploying local assets to mosaik_web...")

    paths = get_paths()

    if not paths["local_html"].exists():
        print(f"❌ Local HTML not found: {paths['local_html']}")
        print("💡 Run with --create to create initial assets")
        return False

    # Create backup if not exists
    if not paths["backup_html"].exists() and paths["target_html"].exists():
        shutil.copy2(paths["target_html"], paths["backup_html"])
        print(f"📋 Backup created: {paths['backup_html']}")

    # Deploy HTML
    shutil.copy2(paths["local_html"], paths["target_html"])
    print(f"📄 HTML deployed: {paths['local_html']} → {paths['target_html']}")

    # Deploy media (if needed)
    if paths["local_media"].exists():
        for media_file in paths["local_media"].iterdir():
            if media_file.is_file():
                target_file = paths["target_media"] / media_file.name
                if (
                    not target_file.exists()
                    or media_file.stat().st_mtime > target_file.stat().st_mtime
                ):
                    shutil.copy2(media_file, target_file)
                    print(f"📁 Media deployed: {media_file.name}")

    print("✅ Local assets deployed successfully!")
    return True


def restore_assets():
    """Restore original mosaik_web assets"""
    print("🔄 Restoring original mosaik_web assets...")

    paths = get_paths()

    if not paths["backup_html"].exists():
        print(f"❌ Backup not found: {paths['backup_html']}")
        return False

    # Restore HTML
    shutil.copy2(paths["backup_html"], paths["target_html"])
    print(f"📄 HTML restored: {paths['backup_html']} → {paths['target_html']}")

    print("✅ Original assets restored successfully!")
    return True


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="WebVis Assets Management")
    parser.add_argument(
        "--create", action="store_true", help="Create initial local assets"
    )
    parser.add_argument(
        "--deploy", action="store_true", help="Deploy local assets to mosaik_web"
    )
    parser.add_argument(
        "--restore", action="store_true", help="Restore original mosaik_web assets"
    )
    parser.add_argument(
        "--status", action="store_true", help="Check current assets status"
    )

    args = parser.parse_args()

    if args.create:
        create_initial_assets()
    elif args.deploy:
        deploy_assets()
    elif args.restore:
        restore_assets()
    elif args.status:
        check_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
