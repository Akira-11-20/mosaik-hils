# WebVis Customization Guide

このガイドでは、mosaik_webのカスタマイズを`uv sync`後でも維持する方法を説明します。

## 🔄 問題: uv syncで変更が消える

`uv sync`や`uv add`を実行すると、`.venv/lib/python3.9/site-packages/mosaik_web/html/index.html`の変更が失われます。

## ✅ 解決策: 自動カスタマイズ

### 方法1: 自動カスタマイズ（推奨）

**main.py起動時に自動適用:**
```bash
uv run python main.py
```
- main.pyが起動時に自動でWebVisをカスタマイズ
- 毎回の起動で最新カスタマイズが適用される

### 方法2: 手動カスタマイズ

**uv sync後に手動実行:**
```bash
uv sync
python customize_webvis.py
```

**元に戻す場合:**
```bash
python customize_webvis.py --restore
```

### 方法3: カスタムWebVisシミュレーター

**main.pyで設定変更:**
```python
"WebVis": {
    "python": "custom_webvis:CustomWebVisSimulator",
},
```

## 🎨 適用されるカスタマイズ

### 1. タイトル変更
```html
<title>mosaik-web (HILS Custom)</title>
```

### 2. 遅延統計エリア
右上に表示される統計パネル:
- 🔄 Delay Node Statistics
- 📊 Packets: 受信/送信数
- ⏱️ Avg Delay: 平均遅延時間
- 📈 Jitter: 最新ジッター値

## 📁 ファイル構成

```
mosaik-hils/
├── customize_webvis.py     # 自動カスタマイズスクリプト
├── custom_webvis.py        # カスタムWebVisシミュレーター
├── main.py                 # 自動カスタマイズ統合済み
└── WEBVIS_CUSTOMIZATION.md # このファイル
```

## 🔧 カスタマイズの追加方法

### HTMLの変更
`customize_webvis.py`の`customize_html()`関数を編集:

```python
def customize_html(html_path):
    content = html_path.read_text()
    
    # 新しいカスタマイズを追加
    content = content.replace(
        "置換対象",
        "新しい内容"
    )
    
    html_path.write_text(content)
```

### CSSの変更
HTMLの`<style>`タグ内にCSSを追加:

```python
css_style = """
<style>
.custom-element {
    color: blue;
    font-weight: bold;
}
</style>
"""

content = content.replace(
    "</head>",
    f"{css_style}</head>"
)
```

### JavaScriptの機能追加
`<script>`タグでJavaScriptを追加:

```python
js_code = """
<script>
// 遅延統計の動的更新
function updateDelayStats(data) {
    document.getElementById('delay-info').textContent = data.info;
}
</script>
"""

content = content.replace(
    "</body>",
    f"{js_code}</body>"
)
```

## 🐛 トラブルシューティング

### カスタマイズが適用されない
1. `python customize_webvis.py` を手動実行
2. ブラウザのキャッシュをクリア (Ctrl+F5)
3. `http://localhost:8002` を再読み込み

### バックアップから復元
```bash
python customize_webvis.py --restore
```

### 動作確認
```bash
uv run python main.py
# http://localhost:8002 にアクセスしてタイトルを確認
```

## 📝 開発ワークフロー

1. **新しい依存関係追加時:**
   ```bash
   uv add new-package
   python customize_webvis.py  # カスタマイズ再適用
   ```

2. **環境再構築時:**
   ```bash
   rm -rf .venv
   uv sync
   python customize_webvis.py  # カスタマイズ適用
   ```

3. **通常の開発時:**
   ```bash
   uv run python main.py  # 自動カスタマイズで起動
   ```

これで`uv sync`後でもWebVisのカスタマイズが維持されます！