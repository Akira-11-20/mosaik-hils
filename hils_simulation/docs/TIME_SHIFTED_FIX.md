# Time-Shifted 接続の修正

## 日時
2025-10-13

## 問題点

以前の実装では、Env → Controller に `time_shifted=True` を使用していましたが、これは不適切でした。

### 以前の接続
```
Controller → Bridge(cmd) → Plant → Bridge(sense) → Env
     ↑                                              |
     └──────────────────────────────────────────────┘
              (time_shifted=True)
```

### 問題
- **Bridge(sense) が既に遅延を持っている**（100ms）
- Env の出力は Bridge(sense) を経由して遅延するため、循環依存は自然に解消される
- つまり、Env → Controller の time-shifted 接続は不要

## 正しい接続

### 修正後の接続
```
Controller → Bridge(cmd) → Plant → Bridge(sense) → Env
     ↑          ↑                                   |
     |          └─ time_shifted=True                |
     └────────────────────────────────────────────────┘
              (通常接続でOK)
```

### 理由
1. **Controller → Bridge(cmd) を time-shifted にする**
   - Controller は時刻 t の状態を読んで、時刻 t+1 の指令を出す
   - これが物理的に正確な因果関係

2. **Env → Controller は通常接続**
   - Bridge(sense) の遅延（100ms）により、Env の出力は既に遅延して Controller に届く
   - 追加の time-shifted は不要

## 実装の変更

### main_hils.py の変更点

#### 1. Controller → Bridge(cmd) 接続（修正）

**変更前:**
```python
# 1. Controller → Bridge(cmd) - 制御指令経路（パッケージ化コマンド）
world.connect(
    controller,
    bridge_cmd,
    ("command", "input"),
)
```

**変更後:**
```python
# 1. Controller → Bridge(cmd) - 制御指令経路（time-shiftedで循環解決）
print("   ⏱️  Using time-shifted connection for Controller → Bridge(cmd)")
world.connect(
    controller,
    bridge_cmd,
    ("command", "input"),
    time_shifted=True,
    initial_data={
        "command": {"thrust": 0.0, "duration": 0.0}
    },
)
```

#### 2. Env → Controller 接続（修正）

**変更前:**
```python
# 5. Env → Controller - 状態フィードバック（time-shifted接続で循環解決）
print("   ⏱️  Using time-shifted connection for Env → Controller")
world.connect(
    spacecraft,
    controller,
    ("position", "velocity"),
    time_shifted=True,
    initial_data={
        "position": 0.0,
        "velocity": 0.0,
    },
)
```

**変更後:**
```python
# 5. Env → Controller - 状態フィードバック（Bridge(sense)の遅延で循環は既に解消済み）
world.connect(
    spacecraft,
    controller,
    ("position", "velocity"),
)
```

## time_shifted の意味

Mosaik の `time_shifted=True` は：
- **「このステップの出力を次のステップで使う」**という意味
- つまり、1 step_size 分の遅延が自然に入る
- Controller の step_size は 10ms なので、10ms 先の指令を出すことになる

## データフロー

### 時系列での動作

```
時刻 t=0:
  Env(0): position=0.0, velocity=0.0 → Controller
  Controller(0): 状態を読んで command を計算
  → time_shifted により command は t=10 で Bridge に送られる

時刻 t=10:
  Controller(0) の command → Bridge(cmd)
  Bridge(cmd): 遅延バッファに格納（50ms + jitter）

時刻 t=60 (約):
  Bridge(cmd): command を Plant に送信
  Plant: 推力を出力 → Bridge(sense)
  Bridge(sense): 遅延バッファに格納（100ms + jitter）

時刻 t=160 (約):
  Bridge(sense): measured_thrust を Env に送信
  Env: 力を受け取り、運動方程式を積分
  Env: position, velocity を Controller に送信（通常接続）

時刻 t=170 (約):
  Controller: 新しい状態で次の command を計算
  → time_shifted により t=180 で Bridge に送られる
```

## 利点

1. **物理的に正確**: 「現在の状態を見て、次の制御周期で指令を出す」という実際のシステムの動作を反映
2. **Bridge の遅延が有効**: 測定経路の遅延（100ms）が正しく作用する
3. **シンプル**: 不要な time-shifted 接続を削除

## テスト結果

### シミュレーション実行
```bash
uv run python hils_simulation/main_hils.py
```

**結果:**
- ✅ シミュレーション正常終了
- ✅ 200 データポイント記録
- ✅ HDF5 ファイル生成成功（49KB）
- ✅ 実行グラフ・データフローグラフ生成成功

### 動作確認
- Controller は 10ms 周期で動作
- Bridge(cmd) は 50ms ± 10ms の遅延
- Bridge(sense) は 100ms ± 20ms の遅延
- 全てのデータが正しく収集される

## 関連ファイル

- [main_hils.py](main_hils.py:145-183) - 接続設定の変更箇所
- [DATA_COLLECTION.md](DATA_COLLECTION.md) - データ収集機能
- [README.md](README.md) - プロジェクト概要

## まとめ

**修正前:**
- ❌ Env → Controller が time-shifted（不適切）
- ❌ Bridge(sense) の遅延が考慮されていない

**修正後:**
- ✅ Controller → Bridge(cmd) が time-shifted（正しい）
- ✅ Env → Controller は通常接続（Bridge の遅延で循環解消）
- ✅ 物理的に妥当な因果関係

これにより、HILS システムの遅延効果をより正確にシミュレートできるようになりました。
