# Mosaik 初学者向けガイド（詳細版）

## 目次
1. [Mosaikとは](#mosaikとは)
2. [基本概念（詳細）](#基本概念詳細)
3. [シミュレーターの作成方法（完全ガイド）](#シミュレーターの作成方法完全ガイド)
4. [シナリオの構築方法](#シナリオの構築方法)
5. [データの接続パターン（全パターン詳解）](#データの接続パターン全パターン詳解)
6. [実践例：このプロジェクトの構成](#実践例このプロジェクトの構成)
7. [デバッグとトラブルシューティング](#デバッグとトラブルシューティング)
8. [パフォーマンスと最適化](#パフォーマンスと最適化)
9. [よくある質問](#よくある質問)
10. [実践的な演習問題](#実践的な演習問題)

---

## Mosaikとは

**Mosaik**は、異なる種類のシミュレーターを統合して共シミュレーション（co-simulation）を実現するPythonフレームワークです。

### 主な特徴
- **モジュール性**: 各シミュレーターを独立して開発・テスト可能
- **言語非依存**: Python以外のシミュレーターも統合可能（コマンドライン経由）
- **時間同期**: 異なる時間ステップのシミュレーターを協調動作
- **スケーラビリティ**: 小規模から大規模システムまで対応

### 適用例
- エネルギーシステムシミュレーション（スマートグリッド等）
- Hardware-in-the-Loop（HILS）シミュレーション
- サイバーフィジカルシステムの検証
- 通信ネットワークと物理システムの統合シミュレーション

---

## 基本概念（詳細）

### 1. World（ワールド）- シミュレーション統合の中核

Worldオブジェクトは、Mosaikにおける「指揮者」の役割を果たします。全シミュレーターのライフサイクル管理、時間同期、データ交換の調整を行います。

#### 初期化と構成

```python
import mosaik

sim_config = {
    # Python製シミュレーター
    "SimulatorA": {
        "python": "path.to.simulator:SimulatorClass",  # モジュールパス:クラス名
        "api_version": "1",  # Mosaik API バージョン（オプション）
    },
    # コマンドライン型シミュレーター（Java, C++等）
    "SimulatorB": {
        "cmd": "java -jar simulator.jar %(addr)s",  # %(addr)s は自動置換
    },
    # 既存のMosaikツール
    "WebVis": {
        "cmd": "mosaik-web %(addr)s --serve=127.0.0.1:9000",
    },
}

world = mosaik.World(
    sim_config,
    debug=True,              # 実行グラフ記録（必須：デバッグ時）
    time_resolution=1.0,      # 1.0 = 1秒、0.001 = 1ミリ秒
    cache=False,              # データキャッシング（通常False推奨）
)
```

#### Worldの主要メソッド

| メソッド | 説明 | 使用例 |
|---------|------|--------|
| `start(name, **params)` | シミュレーター起動 | `sim = world.start("SimA", step_size=10)` |
| `connect(src, dest, attrs)` | エンティティ接続 | `world.connect(e1, e2, ("out", "in"))` |
| `run(until, rt_factor)` | シミュレーション実行 | `world.run(until=3600, rt_factor=1.0)` |
| `shutdown()` | 全シミュレーター終了 | `world.shutdown()` |

#### time_resolutionの重要性

```python
# ミリ秒精度が必要な場合（通信シミュレーション等）
world = mosaik.World(sim_config, time_resolution=0.001)
# この場合、until=1000 は実時間の1秒に相当

# 秒精度で十分な場合（エネルギーシステム等）
world = mosaik.World(sim_config, time_resolution=1.0)
# この場合、until=3600 は実時間の1時間に相当
```

**重要**: time_resolutionは全シミュレーターで共通。途中で変更不可。

---

### 2. Simulator（シミュレーター）- モデルの実装単位

シミュレーターは、特定のドメイン（物理、通信、制御等）のモデルを実装するコンポーネントです。

#### シミュレーター種別の詳細比較

| 種別 | 実行タイミング | step()戻り値 | 適用例 |
|------|--------------|-------------|--------|
| **time-based** | 定期的（step_size毎） | `time + step_size` | 物理シミュレーション、周期的センサー |
| **event-based** | 入力イベント発生時 | `None` | 離散イベント、非周期的制御 |
| **hybrid** | 両方の特性 | 状況依存 | 複雑な制御システム |

#### time-basedシミュレーターの実行フロー

```
時刻 0: step(0, inputs) → 10 を返す
  ↓ Mosaikが10秒待機
時刻 10: step(10, inputs) → 20 を返す
  ↓ Mosaikが10秒待機
時刻 20: step(20, inputs) → 30 を返す
  ↓ 以下繰り返し
```

#### event-basedシミュレーターの実行フロー

```
時刻 0: 起動、待機
  ↓
時刻 5: 外部から入力 → step(5, inputs) → None
  ↓ 再び待機
時刻 23: 外部から入力 → step(23, inputs) → None
  ↓ 入力があるまで実行されない
```

---

### 3. Entity（エンティティ）- モデルのインスタンス

エンティティは、シミュレーター内で作成される個別のモデルインスタンスです。

#### エンティティのライフサイクル

```python
# 1. シミュレーター起動
sim = world.start("NumericalSim", step_size=10)

# 2. エンティティ作成（内部でcreate()が呼ばれる）
entity = sim.NumericalModel(initial_value=1.0, step_size=0.5)

# 3. エンティティの属性アクセス
print(entity.eid)        # 例: "NumSim_0"
print(entity.full_id)    # 例: "NumericalSim-0.NumSim_0"
print(entity.type)       # 例: "NumericalModel"

# 4. 接続設定
world.connect(entity, other_entity, ("output", "input"))

# 5. シミュレーション実行中、エンティティは自動的にstep()される
world.run(until=100)
```

#### エンティティの命名規則

```python
# シミュレーター側での実装
class MySimulator(mosaik_api.Simulator):
    def __init__(self):
        super().__init__(meta)
        self.eid_prefix = "MySim_"  # エンティティID接頭辞
        self.entity_counter = 0

    def create(self, num, model, **params):
        entities = []
        for i in range(num):
            eid = f"{self.eid_prefix}{self.entity_counter}"  # "MySim_0", "MySim_1", ...
            self.entity_counter += 1
            # ...
        return entities
```

---

### 4. Connection（接続）- データフローの定義

接続は、エンティティ間のデータフローを定義します。Mosaikは接続定義に基づいて自動的にデータを転送します。

#### 接続の内部動作

```python
world.connect(entity_a, entity_b, ("output", "input"))
```

この接続により、以下の処理が自動化されます：

```
時刻 T:
1. entity_a.step(T, ...) 実行
2. Mosaikがentity_a.get_data({"entity_a_id": ["output"]}) を呼び出し
3. 取得した "output" 値を保持
4. entity_b.step(T, inputs) を呼び出し
   inputs = {
       "entity_b_id": {
           "input": {
               "entity_a_id": output_value  # ここに自動挿入
           }
       }
   }
```

#### 接続時の属性マッピング

```python
# 同名属性の場合（省略記法）
world.connect(entity_a, entity_b, "data")  # "data" → "data"

# 異なる名前の場合
world.connect(entity_a, entity_b, ("output", "input"))  # "output" → "input"

# 複数属性の接続
world.connect(entity_a, entity_b, ("out1", "out2", "out3"))
# 上記は以下と同等：
# world.connect(entity_a, entity_b, ("out1", "out1"))
# world.connect(entity_a, entity_b, ("out2", "out2"))
# world.connect(entity_a, entity_b, ("out3", "out3"))
```

#### 弱参照接続（weak connection）

```python
# 通常の接続（強参照）: entity_bはentity_aのデータを必ず受信
world.connect(entity_a, entity_b, ("output", "input"))

# 弱参照接続: entity_aがデータを出力しなくても、entity_bは実行される
world.connect(entity_a, entity_b, ("output", "input"), weak=True)
```

**用途**:
- **強参照（デフォルト）**: 制御システム等、データ依存が強い場合
- **弱参照**: モニタリング、ロギング等、データがなくても動作可能な場合

---

## シミュレーターの作成方法（完全ガイド）

シミュレーター作成は、Mosaikの核心です。ここでは実践的な例とともに、各ステップを詳細に解説します。

### ステップ1: メタデータ定義（シミュレーターの仕様書）

メタデータは、シミュレーターの「インターフェース仕様書」です。Mosaikはこの情報を使って、シミュレーター間の互換性をチェックします。

```python
meta = {
    # シミュレータータイプ
    "type": "time-based",  # "event-based", "hybrid" のいずれか

    # API バージョン（オプション）
    "api_version": "3.0",

    # 追加情報（オプション）
    "extra_methods": ["custom_method"],  # カスタムメソッド名

    # モデル定義
    "models": {
        "ModelName": {
            "public": True,  # 外部から利用可能か
            "params": ["param1", "param2"],  # 初期化時のパラメータ
            "attrs": ["attr1", "attr2"],     # 公開属性（接続可能な属性）
            "any_inputs": False,  # 任意の属性を入力として受け入れるか
            "trigger": [],        # このモデルをトリガーする属性リスト
        },
    },
}
```

#### 実践例：センサーシミュレーター

```python
meta = {
    "type": "time-based",
    "models": {
        "TemperatureSensor": {
            "public": True,
            "params": [
                "location",      # センサー設置場所
                "accuracy",      # 精度（±度）
                "sample_rate",   # サンプリング周期（秒）
            ],
            "attrs": [
                "temperature",   # 温度値（℃）
                "timestamp",     # タイムスタンプ
                "status",        # センサー状態（"ok", "error"）
                "battery_level", # バッテリー残量（%）
            ],
        },
        "HumiditySensor": {
            "public": True,
            "params": ["location", "accuracy"],
            "attrs": ["humidity", "timestamp", "status"],
        },
    },
}
```

### ステップ2: Simulatorクラス実装（詳細）

#### 基本構造

```python
import mosaik_api

class MySimulator(mosaik_api.Simulator):
    """
    カスタムシミュレーター実装

    このクラスは mosaik_api.Simulator を継承し、
    Mosaik API の要求するメソッドを実装します。
    """

    def __init__(self):
        """
        コンストラクタ

        注意: このメソッドはシミュレータープロセス起動時に
        　　　1回だけ呼ばれます。init()とは異なります。
        """
        super().__init__(meta)  # 必須: メタデータを渡す

        # シミュレーターレベルの状態管理
        self.entities = {}  # エンティティデータ保存用
        self.step_size = 1  # デフォルトステップサイズ
        self.time = 0       # 現在時刻

    def init(self, sid, time_resolution=1.0, **sim_params):
        """
        シミュレーター初期化（Mosaikから呼ばれる）

        Args:
            sid: シミュレーターID（Mosaikが自動割り当て）
            time_resolution: 時間解像度（Worldの設定値）
            **sim_params: world.start()で渡された追加パラメータ

        Returns:
            dict: メタデータ辞書（self.meta）
        """
        self.sid = sid
        self.time_resolution = time_resolution

        # world.start()で渡されたパラメータを取得
        self.step_size = sim_params.get("step_size", 1)

        print(f"[{self.sid}] Initialized with step_size={self.step_size}")

        return self.meta

    def create(self, num, model, **model_params):
        """
        エンティティ作成（Mosaikから呼ばれる）

        Args:
            num: 作成するエンティティ数
            model: モデルタイプ名（metaで定義したもの）
            **model_params: エンティティ固有のパラメータ

        Returns:
            list: エンティティ情報の辞書リスト
                  [{"eid": "entity_id", "type": "ModelName"}, ...]
        """
        entities = []

        for i in range(num):
            # ユニークなエンティティID生成
            eid = f"{model}_{len(self.entities)}"

            # エンティティの状態データを初期化
            self.entities[eid] = {
                # model_paramsから値を取得（デフォルト値付き）
                "value": model_params.get("initial_value", 0.0),
                "status": "initialized",
                # 内部状態
                "_created_at": self.time,
            }

            entities.append({"eid": eid, "type": model})
            print(f"[{self.sid}] Created entity {eid}")

        return entities

    def step(self, time, inputs, max_advance=None):
        """
        シミュレーションステップ実行（Mosaikから呼ばれる）

        Args:
            time: 現在のシミュレーション時刻
            inputs: 他シミュレーターからの入力データ
                    形式: {eid: {attr: {source_eid: value}}}
            max_advance: 最大進行可能時刻（通常は使用しない）

        Returns:
            int/float: 次にstep()を呼ぶべき時刻
        """
        self.time = time

        # 全エンティティを更新
        for eid, entity in self.entities.items():
            # 1. 入力データの処理
            if eid in inputs:
                for attr, values in inputs[eid].items():
                    # valuesは {source_eid: value} の辞書
                    for source_eid, value in values.items():
                        print(f"[{eid}] Received {attr}={value} from {source_eid}")

                        # 入力データを処理（例：合計を計算）
                        if attr == "input_signal":
                            entity["value"] += value

            # 2. シミュレーションロジック
            # 例: 単純な時間発展
            entity["value"] = entity["value"] * 0.95  # 減衰
            entity["status"] = "active"

        # 次のステップ時刻を返す
        return time + self.step_size

    def get_data(self, outputs):
        """
        エンティティデータの取得（Mosaikから呼ばれる）

        Args:
            outputs: 要求されたデータの仕様
                     形式: {eid: [attr1, attr2, ...]}

        Returns:
            dict: 要求されたデータ
                  形式: {eid: {attr1: value1, attr2: value2}}
        """
        data = {}

        for eid, attrs in outputs.items():
            if eid not in self.entities:
                continue  # 存在しないエンティティはスキップ

            data[eid] = {}
            entity = self.entities[eid]

            for attr in attrs:
                if attr in entity:
                    data[eid][attr] = entity[attr]
                else:
                    # 属性が存在しない場合のデフォルト値
                    data[eid][attr] = None

        return data

    def finalize(self):
        """
        シミュレーション終了処理（Mosaikから呼ばれる）

        ファイル保存、リソース解放等の後処理を実装します。
        """
        print(f"[{self.sid}] Finalizing... Total entities: {len(self.entities)}")

        # 例: 統計情報の出力
        for eid, entity in self.entities.items():
            print(f"  {eid}: final_value={entity['value']:.3f}")

# エントリーポイント
if __name__ == "__main__":
    # このスクリプトが直接実行された場合、シミュレーターとして起動
    mosaik_api.start_simulator(MySimulator())
```

---

### ステップ3: 必須メソッドの詳細解説

#### 1. `init(sid, time_resolution, **sim_params)` - 初期化

**呼び出しタイミング**: `world.start()`実行時

**目的**: シミュレーター全体の初期設定

```python
# シナリオ側
sim = world.start("MySim", step_size=10, custom_param="value")

# シミュレーター側で受け取る
def init(self, sid, time_resolution, step_size=1, custom_param=None):
    self.step_size = step_size          # 10
    self.custom_param = custom_param     # "value"
    return self.meta
```

**返却値**: メタデータ辞書（通常は`self.meta`）

---

#### 2. `create(num, model, **model_params)` - エンティティ作成

**呼び出しタイミング**: `sim.ModelName(...)`実行時

**目的**: モデルインスタンスの作成

```python
# シナリオ側
entity = sim.MyModel(initial_value=100, name="sensor1")

# シミュレーター側で受け取る
def create(self, num, model, initial_value=0, name="default"):
    # num は通常1（明示的に複数作成も可能）
    # model は "MyModel"
    # initial_value, name はエンティティ固有のパラメータ
```

**返却値**: エンティティ情報のリスト

```python
[
    {"eid": "MyModel_0", "type": "MyModel"},
    {"eid": "MyModel_1", "type": "MyModel"},
]
```

---

#### 3. `step(time, inputs, max_advance)` - ステップ実行

**呼び出しタイミング**: シミュレーション実行中、定期的/イベント時

**目的**: シミュレーションロジックの実行

**inputsの構造（重要）**:

```python
inputs = {
    "MyEntity_0": {              # 受信側エンティティID
        "input_attr": {          # 受信側属性名
            "SourceEntity_0": 1.5,   # 送信元ID: 値
            "SourceEntity_1": 2.3,   # 複数ソース可能
        },
        "another_input": {
            "OtherSource_0": 100,
        }
    },
    "MyEntity_1": {
        # ...
    }
}
```

**inputsの処理パターン**:

```python
# パターン1: 単一ソースを想定
def step(self, time, inputs, max_advance=None):
    for eid in inputs:
        for attr, values in inputs[eid].items():
            value = list(values.values())[0]  # 最初の値を取得
            # value を使った処理

# パターン2: 複数ソースの合計
def step(self, time, inputs, max_advance=None):
    for eid in inputs:
        for attr, values in inputs[eid].items():
            total = sum(values.values())  # 全ソースの合計
            # total を使った処理

# パターン3: 個別ソースを区別
def step(self, time, inputs, max_advance=None):
    for eid in inputs:
        for attr, values in inputs[eid].items():
            for source_eid, value in values.items():
                print(f"{source_eid} から {value} を受信")
                # ソース毎に異なる処理
```

**返却値**: 次のstep()呼び出し時刻

- time-based: `time + self.step_size`
- event-based: `None`（次の入力まで待機）

---

#### 4. `get_data(outputs)` - データ取得

**呼び出しタイミング**: 接続された他シミュレーターがデータ要求時

**目的**: 他シミュレーターへのデータ提供

**outputsの構造**:

```python
outputs = {
    "MyEntity_0": ["output", "status"],  # 要求される属性リスト
    "MyEntity_1": ["value"],
}
```

**返却値の構造**:

```python
{
    "MyEntity_0": {
        "output": 1.23,
        "status": "ok",
    },
    "MyEntity_1": {
        "value": 456.78,
    }
}
```

---

### ステップ4: オプションメソッド

#### 1. `finalize()` - 終了処理

```python
def finalize(self):
    """シミュレーション終了時の後処理"""
    # データの保存
    import json
    with open("results.json", "w") as f:
        json.dump(self.entities, f, indent=2)

    # 統計情報の出力
    print(f"Total simulation steps: {self.time}")

    # リソースの解放（ファイル、接続等）
    if hasattr(self, "data_file"):
        self.data_file.close()
```

#### 2. `get_related_entities(entities)` - 関連エンティティ取得

高度な接続パターンで使用。通常は不要。

```python
def get_related_entities(self, entities=None):
    """
    特定のエンティティに関連する他のエンティティを返す

    例: グリッドシミュレーションで、ノードに接続された
    　　全てのエッジを返す等
    """
    related = {}
    for eid in entities or self.entities:
        # 関連エンティティのリストを構築
        related[eid] = [other_eid for other_eid in self.entities if other_eid != eid]
    return related
```

---

## シナリオの構築方法

### 基本的なシナリオ構造

```python
import mosaik

def main():
    # 1. シミュレーター構成
    sim_config = {
        "SimA": {"python": "module.simulator_a:SimulatorA"},
        "SimB": {"python": "module.simulator_b:SimulatorB"},
    }

    # 2. ワールド作成
    world = mosaik.World(sim_config, debug=True)

    # 3. シミュレーター起動
    sim_a = world.start("SimA", step_size=10)
    sim_b = world.start("SimB", step_size=5)

    # 4. エンティティ作成
    entity_a = sim_a.ModelA(param1=100)
    entity_b = sim_b.ModelB(param2="config")

    # 5. 接続設定
    world.connect(entity_a, entity_b, ("output", "input"))

    # 6. シミュレーション実行
    world.run(until=3600)  # 3600秒間実行

if __name__ == "__main__":
    main()
```

### ステップサイズの考え方

```python
# 高速更新が必要なシミュレーター
fast_sim = world.start("FastSim", step_size=1)    # 1秒毎

# 中速更新のシミュレーター
medium_sim = world.start("MediumSim", step_size=10)  # 10秒毎

# 低速更新のシミュレーター
slow_sim = world.start("SlowSim", step_size=60)   # 60秒毎
```

**ポイント**:
- 小さいstep_sizeほど実行頻度が高い
- 必要な精度と計算コストのバランスを考慮
- 異なるstep_sizeのシミュレーターも共存可能

---

## データの接続パターン（全パターン詳解）

データの接続は、Mosaikシミュレーションの要です。ここでは全パターンを実例とともに解説します。

### 1. 一対一接続（One-to-One）- 基本パターン

```python
world.connect(entity_a, entity_b, ("output", "input"))
```

```
[Entity A] --output--> [Entity B]
```

### 2. 多対一接続（Many-to-One）

```python
import mosaik.util

mosaik.util.connect_many_to_one(
    world,
    [entity_a1, entity_a2, entity_a3],  # ソース
    entity_b,                            # ターゲット
    "output"                             # 属性
)
```

```
[Entity A1] --output-->
[Entity A2] --output--> [Entity B]
[Entity A3] --output-->
```

**使用例**: 複数のセンサーデータを1つのデータコレクターに集約

### 3. ランダム接続（Random）

```python
mosaik.util.connect_randomly(
    world,
    [entity_a1, entity_a2],
    [entity_b1, entity_b2],
    ("output", "input")
)
```

**使用例**: ネットワークトポロジーのシミュレーション

### 4. 複数属性の接続

```python
world.connect(
    entity_a,
    entity_b,
    ("attr1", "attr2", "attr3")  # 複数属性を一括接続
)
```

### 5. 非同期接続（async_requests）

```python
world.connect(
    entity_a,
    entity_b,
    ("output", "input"),
    async_requests=True  # 非同期データ要求
)
```

**用途**: 遅延やバッファリングが必要な場合

---

## 実践例：このプロジェクトの構成

### システムアーキテクチャ

```
┌─────────────────┐
│ NumericalModel  │ (正弦波生成)
│   step_size=10  │
└────────┬────────┘
         │ output
         ↓
    ┌────────────┐
    │ DelayNode  │ (通信遅延シミュレーション)
    │ step_size=1│  - 基本遅延: 5秒
    └─────┬──────┘  - ジッター: ±1秒
          │ delayed_output - パケットロス: 1%
          ↓
┌──────────────────────┐
│ HardwareInterface    │ (センサー/アクチュエーター)
│   step_size=10       │
└──────────────────────┘
         │
         ↓ (全データ)
┌──────────────────────┐
│  DataCollector       │ (HDF5保存)
│  WebVis             │ (リアルタイム可視化)
└──────────────────────┘
```

### 実装のポイント

#### 1. 遅延ノードの実装 ([delay_simulator.py](../src/simulators/delay_simulator.py))

```python
class DelayNode:
    def __init__(self, base_delay=5.0, jitter_std=1.0, packet_loss_rate=0.01):
        self.base_delay = base_delay
        self.jitter_std = jitter_std
        self.packet_loss_rate = packet_loss_rate
        self.packet_buffer = []  # (arrival_time, data, output_time, seq_num)

    def step(self, current_time, inputs):
        # 1. 入力データ受信
        if inputs:
            jitter = random.gauss(0, self.jitter_std)
            output_time = current_time + self.base_delay + jitter

            # パケットロス判定
            if random.random() >= self.packet_loss_rate:
                self.packet_buffer.append((current_time, data, output_time, seq))

        # 2. 出力準備完了パケットの処理
        ready_packets = [p for p in self.packet_buffer if p[2] <= current_time]
        if ready_packets:
            self.current_output = ready_packets[0][1]
            # バッファから削除
```

**特徴**:
- ガウシアンジッターで通信の不確実性を再現
- パケットバッファで遅延管理
- 統計情報（平均遅延、パケット損失数）を提供

#### 2. データ収集の実装 ([data_collector.py](../src/simulators/data_collector.py))

```python
def step(self, time, inputs, max_advance=None):
    for eid in inputs:
        data_point = {"time": time}
        for attr, values in inputs[eid].items():
            for source_eid, value in values.items():
                data_point[f"{attr}_{source_eid}"] = value

        self.data_log.append(data_point)

    return time + self.step_size

def finalize(self):
    """シミュレーション終了時にHDF5保存"""
    with h5py.File(output_path, "w") as h5_file:
        steps_group = h5_file.create_group("steps")
        for key in keys:
            column = [entry.get(key) for entry in self.data_log]
            steps_group.create_dataset(name=key, data=column)
```

**特徴**:
- リアルタイムコンソール出力
- HDF5形式で高速保存
- 複数ソースからのデータを統合

#### 3. メインシナリオ ([main.py](../main.py))

```python
# シミュレーター起動（異なるstep_size）
numerical_sim = world.start("NumericalSim", step_size=10)
hardware_sim = world.start("HardwareSim", step_size=10)
delay_sim = world.start("DelaySim", step_size=1)  # 高頻度実行

# エンティティ作成
numerical_model = numerical_sim.NumericalModel(initial_value=1.0)
hardware_interface = hardware_sim.HardwareInterface(device_id="sensor_01")
delay_node = delay_sim.DelayNode(
    base_delay=5,
    jitter_std=1,
    packet_loss_rate=0.01
)

# データフロー接続
world.connect(numerical_model, delay_node, ("output", "input"))
world.connect(delay_node, hardware_interface, ("delayed_output", "actuator_command"))

# データ収集設定
mosaik.util.connect_many_to_one(world, [numerical_model], collector, "output")
mosaik.util.connect_many_to_one(world, [delay_node], collector, "delayed_output", "stats")
mosaik.util.connect_many_to_one(world, [hardware_interface], collector, "sensor_value")

# 実行
world.run(until=30, rt_factor=1)
```

---

## デバッグとトラブルシューティング

### デバッグ手法の総まとめ

#### 1. ログ出力戦略

```python
import logging

class MySimulator(mosaik_api.Simulator):
    def __init__(self):
        super().__init__(meta)
        # ロガーの設定
        self.logger = logging.getLogger(f"MySim")
        self.logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler("simulator.log")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def step(self, time, inputs, max_advance=None):
        self.logger.debug(f"Step at time={time}")
        self.logger.debug(f"Received inputs: {inputs}")

        # ... シミュレーションロジック ...

        self.logger.info(f"Processed {len(inputs)} entities")
        return time + self.step_size
```

#### 2. データ検証

```python
def step(self, time, inputs, max_advance=None):
    # 入力データの検証
    for eid in inputs:
        for attr, values in inputs[eid].items():
            for source_eid, value in values.items():
                # 異常値チェック
                if not isinstance(value, (int, float)):
                    print(f"WARNING: Invalid type from {source_eid}: {type(value)}")
                elif abs(value) > 1000:
                    print(f"WARNING: Outlier detected from {source_eid}: {value}")

    return time + self.step_size
```

#### 3. Mosaikデバッグモードの活用

```python
# main.py
world = mosaik.World(sim_config, debug=True)  # 実行グラフ記録を有効化

world.run(until=100)

# 実行後、グラフを生成
import mosaik.util

mosaik.util.plot_dataflow_graph(world, folder="./debug")
mosaik.util.plot_execution_graph(world, folder="./debug")
mosaik.util.plot_execution_time(world, folder="./debug")
```

生成されるグラフ:
- `dataflow_graph.pdf`: エンティティ間のデータフロー
- `execution_graph.pdf`: 実行順序と依存関係
- `execution_time.pdf`: 各シミュレーターの実行時間

#### 4. よくあるエラーと解決方法

| エラーメッセージ | 原因 | 解決方法 |
|----------------|------|---------|
| `AttributeError: 'NoneType' object has no attribute...` | get_data()でNoneを返している | メタデータに定義された全属性を返す |
| `KeyError: 'eid'` | create()の返却値が不正 | `[{"eid": "...", "type": "..."}]`形式を確認 |
| `AssertionError: Step size must be > 0` | step()が負の値を返した | `time + self.step_size`が正であることを確認 |
| `Connection failed: no such attribute` | メタデータに属性が未定義 | `attrs`リストに属性を追加 |

#### 5. ステップバイステップデバッグ

```python
def step(self, time, inputs, max_advance=None):
    print(f"\n{'='*50}")
    print(f"STEP at time={time}")
    print(f"{'='*50}")

    # 入力の詳細表示
    if inputs:
        print("INPUTS:")
        for eid, attrs_dict in inputs.items():
            print(f"  Entity: {eid}")
            for attr, sources in attrs_dict.items():
                print(f"    Attribute: {attr}")
                for src_eid, value in sources.items():
                    print(f"      From {src_eid}: {value}")
    else:
        print("No inputs received")

    # ... 処理 ...

    print(f"Next step: {time + self.step_size}\n")
    return time + self.step_size
```

---

## パフォーマンスと最適化

### 1. ステップサイズの最適化

```python
# ❌ 悪い例: 全シミュレーターが高頻度
world.start("SimA", step_size=1)  # 毎秒実行
world.start("SimB", step_size=1)  # 毎秒実行
world.start("SimC", step_size=1)  # 毎秒実行
# 結果: 大量の不要な計算

# ✅ 良い例: 必要性に応じたステップサイズ
world.start("ControlSim", step_size=1)      # 制御: 高頻度必要
world.start("PhysicsSim", step_size=10)     # 物理: 中頻度で十分
world.start("MonitoringSim", step_size=60)  # 監視: 低頻度で十分
```

### 2. データ収集の効率化

```python
# ❌ 悪い例: 毎ステップファイル書き込み
def step(self, time, inputs, max_advance=None):
    with open("data.csv", "a") as f:  # 遅い！
        f.write(f"{time},{self.value}\n")
    return time + self.step_size

# ✅ 良い例: メモリに蓄積、finalize()で一括保存
def step(self, time, inputs, max_advance=None):
    self.data_buffer.append((time, self.value))  # 高速
    return time + self.step_size

def finalize(self):
    # 終了時に一括保存
    with open("data.csv", "w") as f:
        for time, value in self.data_buffer:
            f.write(f"{time},{value}\n")
```

### 3. 接続パターンの最適化

```python
# ❌ 悪い例: 多数の個別接続
for entity_a in entities_a:
    for entity_b in entities_b:
        world.connect(entity_a, entity_b, "output")
# 結果: O(n²) の接続数

# ✅ 良い例: データコレクターパターン
collector = data_collector.Collector()
mosaik.util.connect_many_to_one(world, entities_a, collector, "output")
mosaik.util.connect_many_to_one(world, entities_b, collector, "output")
# 結果: O(n) の接続数
```

### 4. メモリ使用量の削減

```python
class EfficientSimulator(mosaik_api.Simulator):
    def step(self, time, inputs, max_advance=None):
        # ❌ 悪い例: 全履歴を保存
        # for eid in self.entities:
        #     self.entities[eid]["history"].append(value)  # メモリ増大

        # ✅ 良い例: 必要な情報のみ保持
        for eid in self.entities:
            # 直近の値のみ保持
            self.entities[eid]["current_value"] = value
            # 統計情報のみ更新
            self.entities[eid]["sum"] += value
            self.entities[eid]["count"] += 1

        return time + self.step_size
```

### 5. リアルタイム実行のチューニング

```python
# Hardware-in-the-Loop (HILS) の場合
world.run(until=3600, rt_factor=1.0)  # 実時間と同期

# rt_factorの調整指針:
# - rt_factor < 1.0: シミュレーションが実時間より高速な場合に減速
# - rt_factor = 1.0: 実時間と同期（推奨）
# - rt_factor > 1.0: シミュレーションが実時間より低速な場合に加速（注意）

# シミュレーションが実時間に追いつかない場合の対策:
# 1. ステップサイズを大きくする
# 2. シミュレーションロジックを最適化
# 3. rt_factorを使わず最速実行
```

---

## よくある質問

### Q1: シミュレーターのstep_sizeはどう決めるべきか？

**A**: 以下の要素を考慮します：

1. **物理的な制約**: センサーのサンプリング周期、制御周期など
2. **精度要件**: 高精度が必要な場合は小さいstep_size
3. **計算コスト**: 小さすぎると実行時間が増加
4. **他のシミュレーターとの同期**: データ交換頻度を考慮

**推奨アプローチ**:
```python
# 高精度制御: 1秒
control_sim = world.start("ControlSim", step_size=1)

# 通常のシミュレーション: 10秒
physics_sim = world.start("PhysicsSim", step_size=10)

# 低頻度更新: 60秒以上
monitoring_sim = world.start("MonitorSim", step_size=60)
```

### Q2: inputsの構造が分かりにくい

**A**: `step()`メソッドの`inputs`は以下の構造です：

```python
inputs = {
    "Entity_ID": {
        "attribute_name": {
            "source_Entity_ID": value,
            "another_source_ID": value,
        }
    }
}
```

**例**:
```python
def step(self, time, inputs, max_advance=None):
    for eid, entity in self.entities.items():
        if eid in inputs:
            for attr, values in inputs[eid].items():
                # 複数ソースからの値を処理
                for source_id, value in values.items():
                    print(f"{source_id}から{attr}={value}を受信")

                # 単一値を取得する場合
                single_value = list(values.values())[0]
```

### Q3: エンティティ間の接続でエラーが出る

**A**: よくある原因：

1. **属性名の不一致**
   ```python
   # NG: 送信側の"output"と受信側の"input"が定義されていない
   world.connect(entity_a, entity_b, ("output", "input"))
   ```

   **解決**: メタデータの`attrs`に属性を追加
   ```python
   meta = {
       "models": {
           "Model": {
               "attrs": ["output", "input"],  # 必要な属性を全て列挙
           }
       }
   }
   ```

2. **シミュレータータイプの不一致**
   - event-basedシミュレーターは`async_requests=True`が必要

3. **エンティティが存在しない**
   - `create()`で正しくエンティティを作成したか確認

### Q4: デバッグ方法は？

**A**: 以下の手法を活用してください：

1. **デバッグモード有効化**
   ```python
   world = mosaik.World(sim_config, debug=True)
   ```

2. **ログ出力**
   ```python
   def step(self, time, inputs, max_advance=None):
       print(f"Time {time}: Received inputs: {inputs}")
       return time + self.step_size
   ```

3. **実行グラフの可視化**
   ```python
   import mosaik.util

   world.run(until=100)

   mosaik.util.plot_dataflow_graph(world, folder="./logs")
   mosaik.util.plot_execution_graph(world, folder="./logs")
   mosaik.util.plot_execution_time(world, folder="./logs")
   ```

4. **HDF5データの確認**
   ```python
   import h5py

   with h5py.File("logs/simulation_data.h5", "r") as f:
       print(list(f["steps"].keys()))
       data = f["steps"]["output_NumSim_0"][:]
       print(data)
   ```

### Q5: リアルタイム実行とは？

**A**: `rt_factor`パラメータで制御します：

```python
# リアルタイム同期（1倍速）
world.run(until=3600, rt_factor=1.0)

# 実時間の2倍速（ファスト実行）
world.run(until=3600, rt_factor=2.0)

# 実時間の半分速（スロー実行、デバッグ用）
world.run(until=3600, rt_factor=0.5)

# 最速実行（rt_factorなし）
world.run(until=3600)
```

**用途**:
- `rt_factor=1.0`: Hardware-in-the-Loopテスト
- `rt_factor=None`: 大規模シミュレーション（最速実行）

### Q6: 外部プログラム（非Python）の統合方法は？

**A**: コマンドライン型シミュレーターとして統合できます：

```python
sim_config = {
    "ExternalSim": {
        "cmd": "java -jar simulator.jar %(addr)s",
        # または
        # "cmd": "./my_cpp_simulator %(addr)s",
    }
}
```

**要件**:
- Mosaik API仕様に従った通信プロトコル実装
- 標準入出力またはソケット通信

---

## 次のステップ

### 学習リソース

1. **公式ドキュメント**: https://mosaik.readthedocs.io/
2. **チュートリアル**: https://mosaik.readthedocs.io/en/latest/tutorials/
3. **サンプルコード**: https://gitlab.com/mosaik/examples

### このプロジェクトでの実験アイデア

1. **遅延パラメータの変更**
   ```python
   # main.pyで変更
   COMMUNICATION_DELAY = 10  # 5秒 → 10秒に変更
   JITTER_STD = 2            # ジッターを増加
   ```

2. **新しいシミュレーターの追加**
   - 制御アルゴリズムシミュレーター（PID制御など）
   - ネットワークトラフィックシミュレーター

3. **複数デバイスのシミュレーション**
   ```python
   # 複数のハードウェアインターフェースを作成
   hw_list = hardware_sim.HardwareInterface.create(3, device_id="multi")
   ```

4. **カスタム可視化の追加**
   - WebVisのエンティティタイプ設定をカスタマイズ
   - 独自のプロットスクリプト作成

---

---

## 実践的な演習問題

理解を深めるため、以下の演習問題に取り組んでみましょう。

### 演習1: シンプルなカウンターシミュレーター

**目標**: 基本的なシミュレーターを作成する

**要件**:
1. 毎ステップでカウントアップする `Counter` モデルを作成
2. 初期値とステップ幅をパラメータとして受け取る
3. 現在のカウント値を `count` 属性として公開

**ヒント**:
```python
meta = {
    "type": "time-based",
    "models": {
        "Counter": {
            "public": True,
            "params": ["init_value", "increment"],
            "attrs": ["count"],
        }
    }
}
```

**確認方法**:
```python
# main.py
sim = world.start("CounterSim", step_size=1)
counter = sim.Counter(init_value=0, increment=1)
world.run(until=10)
# 期待結果: 0, 1, 2, ..., 10
```

---

### 演習2: 温度センサーと制御システム

**目標**: 2つのシミュレーターを接続する

**要件**:
1. **TempSensorSimulator**: ランダムな温度（20～30℃）を生成
2. **ControllerSimulator**: 温度が25℃を超えたら冷却指令を出力
3. センサーと制御器を接続

**データフロー**:
```
TempSensor --temperature--> Controller --cooling_command--> (出力)
```

**ヒント**:
```python
# センサー側
import random
entity["temperature"] = random.uniform(20, 30)

# 制御器側
if inputs[eid]["temperature"][source_eid] > 25:
    entity["cooling_command"] = 1  # ON
else:
    entity["cooling_command"] = 0  # OFF
```

---

### 演習3: 遅延パラメータの実験

**目標**: このプロジェクトの遅延シミュレーターを改造する

**タスク**:
1. `main.py` の遅延パラメータを変更
   ```python
   COMMUNICATION_DELAY = 10  # 5秒 → 10秒
   JITTER_STD = 2            # 1秒 → 2秒
   PACKET_LOSS_RATE = 0.05   # 1% → 5%
   ```

2. シミュレーションを実行し、HDF5ファイルを確認

3. 遅延の影響を可視化
   ```python
   import h5py
   import matplotlib.pyplot as plt

   with h5py.File("logs/.../simulation_data.h5", "r") as f:
       time = f["steps"]["time"][:]
       output = f["steps"]["output_NumSim_0"][:]
       delayed = f["steps"]["delayed_output_DelayNode_0"][:]

   plt.plot(time, output, label="Original")
   plt.plot(time, delayed, label="Delayed")
   plt.legend()
   plt.show()
   ```

---

### 演習4: 複数エンティティの管理

**目標**: 複数のセンサーを同時にシミュレート

**要件**:
1. 3つの温度センサーを作成（それぞれ異なる位置）
2. 1つのデータコレクターに全データを集約
3. 各センサーの平均温度を計算

**実装例**:
```python
# シナリオ
sensor_sim = world.start("TempSim", step_size=10)
sensors = [
    sensor_sim.TempSensor(location="room1"),
    sensor_sim.TempSensor(location="room2"),
    sensor_sim.TempSensor(location="room3"),
]

collector = data_sim.Collector()
mosaik.util.connect_many_to_one(world, sensors, collector, "temperature")
```

---

### 演習5: イベント駆動シミュレーター

**目標**: event-basedシミュレーターを作成する

**要件**:
1. 閾値を超えたときのみ動作するアラームシミュレーター
2. 温度センサーからの入力を監視
3. 25℃を超えたらアラートを出力

**実装のポイント**:
```python
meta = {
    "type": "event-based",  # ← 重要
    "models": {
        "Alarm": {
            "public": True,
            "params": ["threshold"],
            "attrs": ["alert_status"],
        }
    }
}

def step(self, time, inputs, max_advance=None):
    for eid in inputs:
        if "temperature" in inputs[eid]:
            temp = list(inputs[eid]["temperature"].values())[0]
            if temp > self.entities[eid]["threshold"]:
                self.entities[eid]["alert_status"] = "ALERT"
                print(f"⚠️  ALERT: Temperature {temp}℃ exceeded threshold!")
            else:
                self.entities[eid]["alert_status"] = "OK"

    return None  # event-basedは None を返す
```

**接続**:
```python
world.connect(sensor, alarm, ("temperature", "temperature"), async_requests=True)
```

---

### 演習6: カスタムデータ可視化

**目標**: シミュレーション結果をグラフ化する

**要件**:
1. HDF5ファイルからデータを読み込み
2. 時系列グラフを作成
3. 複数の属性を同時プロット

**実装例**:
```python
import h5py
import matplotlib.pyplot as plt

def plot_simulation_results(h5_file_path):
    with h5py.File(h5_file_path, "r") as f:
        # データ読み込み
        time = f["steps"]["time"][:]
        output = f["steps"]["output_NumSim_0"][:]
        sensor = f["steps"]["sensor_value_HW_0"][:]

        # プロット
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.plot(time, output, label="Numerical Output")
        ax1.set_ylabel("Output Value")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(time, sensor, label="Sensor Value", color="orange")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Sensor Value (V)")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig("simulation_results.png", dpi=300)
        plt.show()

# 使用
plot_simulation_results("logs/20241013-120000/simulation_data.h5")
```

---

### 演習7: PID制御シミュレーター（上級）

**目標**: 本格的な制御アルゴリズムを実装

**要件**:
1. PID制御器シミュレーターを作成
2. センサーからのフィードバックを受信
3. 目標値に追従する制御信号を出力

**PID制御の実装**:
```python
class PIDController:
    def __init__(self, kp=1.0, ki=0.1, kd=0.05, setpoint=25.0):
        self.kp = kp  # 比例ゲイン
        self.ki = ki  # 積分ゲイン
        self.kd = kd  # 微分ゲイン
        self.setpoint = setpoint  # 目標値

        self.integral = 0
        self.prev_error = 0

    def update(self, measured_value, dt):
        error = self.setpoint - measured_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error

        return output

# シミュレーターに組み込み
def step(self, time, inputs, max_advance=None):
    for eid, entity in self.entities.items():
        if eid in inputs and "temperature" in inputs[eid]:
            temp = list(inputs[eid]["temperature"].values())[0]
            dt = time - entity["last_time"]

            control_signal = entity["pid"].update(temp, dt)
            entity["control_output"] = control_signal
            entity["last_time"] = time

    return time + self.step_size
```

---

## まとめ

Mosaikの基本は以下の4ステップです：

1. **シミュレーター作成**: `mosaik_api.Simulator`を継承
2. **メタデータ定義**: モデルと属性を明示
3. **シナリオ構築**: Worldで統合し接続
4. **実行と分析**: データ収集と可視化

### 学習の進め方

1. **基礎を固める**: 演習1～3で基本操作をマスター
2. **実践力をつける**: 演習4～5で複雑なシステムに挑戦
3. **応用に進む**: 演習6～7で実用的なスキルを習得
4. **独自開発**: このプロジェクトをベースに改造・拡張

### 次のステップ

- **公式ドキュメント**: https://mosaik.readthedocs.io/
- **GitLabサンプル**: https://gitlab.com/mosaik/examples
- **論文**: Mosaik の設計思想と応用例を学ぶ
- **コミュニティ**: Stack Overflow, GitHub Issues で質問

このプロジェクトのコードを参考に、独自のシミュレーションを構築してみてください！

---

**最終更新**: 2024年10月
**対象バージョン**: Mosaik 3.x, mosaik-api 3.x
