# コマンドパッケージ化の実装

## 概要

制御指令（`thrust_cmd`と`duration_cmd`）を**JSON/辞書形式でパッケージ化**し、Bridge経由で送信する実装に変更しました。

## 変更前と変更後

### 変更前（問題あり）

```
Controller ─┬─ thrust_cmd ──→ Bridge(cmd) ──→ Plant
            └─ duration_cmd ─────────────────→ Plant (直接接続)
```

**問題点**:
- `duration_cmd`がBridge経由でなく直接接続
- データフローグラフで Controller → Plant の直接線が表示される
- 遅延の影響が`thrust_cmd`にしか適用されない

### 変更後（改善版）

```
Controller ── command ──→ Bridge(cmd) ──→ Plant
              {thrust, duration}
```

**改善点**:
- ✅ 全ての制御パラメータがBridge経由で送信
- ✅ データフローグラフが正確
- ✅ 遅延が全コマンドに均一に適用される
- ✅ 将来的な拡張が容易（パラメータ追加が簡単）

## データ構造

### Controllerの出力

```python
command = {
    "thrust": 20.0,    # 推力指令 [N]
    "duration": 10,    # 持続時間 [ms]
}
```

### Bridge経由の伝送

```python
# Controller側
world.connect(controller, bridge_cmd, ("command", "input"))

# Plant側
world.connect(bridge_cmd, plant, ("delayed_output", "command"))
```

Bridgeは`command`を**そのまま（透過的に）伝送**します。辞書であっても問題なく扱えます。

### Plantでの受信

```python
def step(self, time, inputs, max_advance=None):
    if eid in inputs and "command" in inputs[eid]:
        cmd = list(inputs[eid]["command"].values())[0]

        if cmd is not None and isinstance(cmd, dict):
            thrust = cmd.get("thrust", 0.0)
            duration = cmd.get("duration", 0.0)
            # コマンドを実行
```

## 実装の詳細

### 1. ControllerSimの変更

**メタデータ**:
```python
"attrs": [
    "position",        # 入力
    "velocity",        # 入力
    "command",         # 出力: パッケージ化コマンド
    "error",           # 出力
],
```

**step()メソッド**:
```python
# コマンドをパッケージ化
entity["command"] = {
    "thrust": thrust,
    "duration": entity["thrust_duration"],
}
```

### 2. PlantSimの変更

**メタデータ**:
```python
"attrs": [
    "command",         # 入力: パッケージ化コマンド
    "measured_thrust", # 出力
    "status",          # 出力
],
```

**step()メソッド**:
```python
if eid in inputs and "command" in inputs[eid]:
    cmd = list(inputs[eid]["command"].values())[0]

    if cmd is not None and isinstance(cmd, dict):
        thrust = cmd.get("thrust", 0.0)
        duration = cmd.get("duration", 0.0)
        # 処理...
```

### 3. main_hils.pyの変更

**接続の簡略化**:
```python
# 変更前: 2本の接続が必要
world.connect(controller, bridge_cmd, ("thrust_cmd", "input"))
world.connect(controller, plant, ("duration_cmd", "duration_cmd"))  # 直接接続

# 変更後: 1本の接続のみ
world.connect(controller, bridge_cmd, ("command", "input"))
world.connect(bridge_cmd, plant, ("delayed_output", "command"))
```

## 利点

### 1. 拡張性

新しいパラメータを追加する場合：

```python
# 容易に拡張可能
command = {
    "thrust": 20.0,
    "duration": 10,
    "mode": "continuous",     # 追加
    "priority": "high",       # 追加
}
```

メタデータや接続を変更する必要なし！

### 2. 型安全性

```python
# 型チェックが容易
if isinstance(cmd, dict):
    thrust = cmd.get("thrust", 0.0)  # デフォルト値も設定可能
    duration = cmd.get("duration", 0.0)
```

### 3. デバッグのしやすさ

```python
# ログ出力が分かりやすい
print(f"Received command: {cmd}")
# 出力例: Received command: {'thrust': 20.0, 'duration': 10}
```

### 4. 実機統合時の利便性

将来、C++やROS等の実機制御器と統合する場合：

```python
# JSON形式でシリアライズ可能
import json
json_cmd = json.dumps(command)
# '{"thrust": 20.0, "duration": 10}'

# ソケット通信等で送信
sock.send(json_cmd.encode())
```

## テスト結果

### 動作確認

```bash
cd hils_simulation
uv run python main_hils.py
```

**出力例**:
```
🔗 Connecting data flows...

✅ Data flow configured:
   Controller → Bridge(cmd) → Plant → Bridge(sense) → Env
   Env → Controller (time-shifted)
   ℹ️  Command format: JSON/dict {thrust, duration}

▶️  Running simulation until 500ms (0.5s)...
======================================================================
[ControllerSim] t=0ms: pos=0.000m, vel=0.000m/s, error=10.000m, thrust=20.000N
[PlantSim] ThrustStand_0: Thrust 20.000N for 10ms
...
======================================================================
✅ Simulation completed successfully!
```

### データフローグラフ

新しいグラフでは：
- ✅ Controller → Bridge(cmd) の接続のみ
- ✅ Controller → Plant の直接接続が**消えた**
- ✅ 全データがBridge経由で流れている

## 今後の拡張案

### 1. 6DOF版への対応

```python
command = {
    "force": [Fx, Fy, Fz],      # 3軸推力
    "torque": [Mx, My, Mz],     # 3軸トルク
    "duration": 10,
}
```

### 2. 複数スラスタ制御

```python
command = {
    "thrusters": {
        "thruster_1": 5.0,
        "thruster_2": 3.0,
        "thruster_3": 2.0,
    },
    "duration": 10,
}
```

### 3. メタデータ付きコマンド

```python
command = {
    "thrust": 20.0,
    "duration": 10,
    "timestamp": 12345,         # 送信時刻
    "sequence_number": 42,      # シーケンス番号
    "priority": "high",         # 優先度
}
```

## まとめ

- ✅ コマンドをJSON/辞書でパッケージ化
- ✅ 全データがBridge経由で送信
- ✅ データフローが正確に表現される
- ✅ 将来の拡張が容易
- ✅ 実機統合時の互換性が向上

**推奨**: 今後の実装では、複数パラメータを送信する場合は常にパッケージ化する方針が良い。
