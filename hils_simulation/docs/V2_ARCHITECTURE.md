# HILS Simulation Architecture

このドキュメントでは、HILS Simulationのシナリオベースアーキテクチャについて説明します。

## 概要

既存の`main_*.py`ファイルで重複していたコードを整理し、シナリオベースのアーキテクチャに移行しました。これにより、コードの保守性と拡張性が大幅に向上しています。

## ディレクトリ構造

```
hils_simulation/
├── config/                          # 設定管理モジュール
│   ├── __init__.py
│   ├── parameters.py                # パラメータ管理（環境変数読み込み、JSON保存）
│   └── sim_config.py                # シミュレーター設定
│
├── scenarios/                       # シナリオ実装モジュール
│   ├── __init__.py
│   ├── base_scenario.py            # 基底クラス（共通機能）
│   ├── hils_scenario.py            # HILSシナリオ（通信遅延あり）
│   ├── rt_scenario.py              # RTシナリオ（通信遅延なし）
│   ├── inverse_comp_scenario.py    # 逆補償シナリオ
│   └── pure_python_scenario.py     # Pure Pythonシナリオ
│
├── main.py                          # 統一エントリーポイント
│
└── archive/                         # v1実装のアーカイブ
    ├── main_hils.py
    ├── main_hils_rt.py
    ├── main_hils_with_inverse_comp.py
    └── main_pure_python.py
```

## 主要コンポーネント

### 1. config/parameters.py

**役割**: 全シミュレーションパラメータの一元管理

**主要クラス**:
- `SimulationParameters`: 全パラメータを集約するメインクラス
- `CommunicationParams`: 通信遅延パラメータ
- `ControlParams`: 制御パラメータ
- `SimulatorParams`: シミュレータ実行周期
- `SpacecraftParams`: 宇宙機物理パラメータ
- `InverseCompParams`: 逆補償パラメータ

**主要機能**:
- 環境変数からのパラメータ読み込み（`.env`ファイル対応）
- パラメータのJSON形式保存
- ステップ数計算などの便利プロパティ

**使用例**:
```python
from config.parameters import SimulationParameters

# 環境変数から読み込み
params = SimulationParameters.from_env()

# パラメータへのアクセス
print(f"Control period: {params.control.control_period}ms")
print(f"Total steps: {params.simulation_steps}")

# JSON保存
params.save_to_json(output_dir, "HILS")
```

### 2. config/sim_config.py

**役割**: Mosaikシミュレーター設定の生成

**主要機能**:
- `get_simulator_config()`: シナリオに応じた設定辞書を返す
  - `include_bridge`: 通信ブリッジの有無
  - `include_inverse_comp`: 逆補償器の有無

### 3. scenarios/base_scenario.py

**役割**: 全シナリオの基底クラス（抽象クラス）

**主要メソッド**:
- `scenario_name`: シナリオ名（抽象プロパティ）
- `scenario_description`: シナリオの説明（抽象プロパティ）
- `create_world()`: Mosaikワールドの作成（抽象メソッド）
- `setup_entities()`: エンティティ作成（抽象メソッド）
- `connect_entities()`: エンティティ接続（抽象メソッド）
- `setup_data_collection()`: データ収集設定
- `generate_graphs()`: グラフ生成
- `run()`: シミュレーション実行（テンプレートメソッド）

**設計パターン**: Template Methodパターン

### 4. scenarios/*_scenario.py

各シナリオクラスは`BaseScenario`を継承し、必要なメソッドを実装します。

#### HILSScenario ([hils_scenario.py](hils_simulation/scenarios/hils_scenario.py))
- 通信遅延とブリッジを含む完全なHILS構成
- cmd/sense両方の経路で遅延を実装
- 結果は`results/`に保存

#### RTScenario ([rt_scenario.py](hils_simulation/scenarios/rt_scenario.py))
- 通信ブリッジなし（直接接続）
- 遅延なしのベースライン比較用
- 結果は`results_rt/`に保存

#### InverseCompScenario ([inverse_comp_scenario.py](hils_simulation/scenarios/inverse_comp_scenario.py))
- HILS + 逆補償器
- コマンド経路に補償器を挿入
- 結果は`results/YYYYMMDD-HHMMSS_inverse_comp/`に保存

#### PurePythonScenario ([pure_python_scenario.py](hils_simulation/scenarios/pure_python_scenario.py))
- Mosaikフレームワークなし
- 純粋なPythonシミュレーション
- 結果は`results_pure/`に保存

### 5. main.py

**役割**: 統一エントリーポイント

**使用方法**:
```bash
# ヘルプ表示
python main.py --help

# HILSシナリオ実行（デフォルト）
python main.py
python main.py hils

# RTシナリオ実行
python main.py rt

# 逆補償シナリオ実行
python main.py inverse_comp

# Pure Pythonシナリオ実行
python main.py pure_python
```

## v1 vs v2 の比較

### v1（従来の実装）

```
hils_simulation/
├── main_hils.py                 # ~400行、多くの重複コード
├── main_hils_rt.py              # ~325行、main_hilsと80%重複
├── main_hils_with_inverse_comp.py  # ~450行、main_hilsと90%重複
└── main_pure_python.py          # ~335行、独自実装
```

**問題点**:
- パラメータ読み込みロジックが4ファイルで重複
- シミュレーション設定保存ロジックが4ファイルで重複
- グラフ生成ロジックが3ファイルで重複
- 新しいパラメータ追加時に4ファイル全てを修正が必要
- コードの保守性が低い

### v2（新しい実装）

```
hils_simulation/
├── config/
│   ├── parameters.py            # パラメータ管理を一元化
│   └── sim_config.py            # シミュレーター設定を一元化
├── scenarios/
│   ├── base_scenario.py         # 共通ロジックを基底クラスに集約
│   ├── hils_scenario.py         # ~150行、差分のみ実装
│   ├── rt_scenario.py           # ~130行、差分のみ実装
│   ├── inverse_comp_scenario.py # ~180行、差分のみ実装
│   └── pure_python_scenario.py  # ~200行、差分のみ実装
└── main_v2.py                   # ~100行、シンプルなCLI
```

**改善点**:
- ✅ コードの重複を90%削減
- ✅ 新しいパラメータの追加が1ファイルで完結
- ✅ 新しいシナリオの追加が容易（基底クラスを継承するだけ）
- ✅ テストが容易（各クラスを独立してテスト可能）
- ✅ 保守性・拡張性の大幅向上

## 新しいシナリオの追加方法

1. `scenarios/`に新しいファイルを作成
2. `BaseScenario`を継承
3. 必要な抽象メソッドを実装
4. `scenarios/__init__.py`にエクスポート追加
5. `main_v2.py`の`scenario_map`に追加

**例**: ジッタ補償シナリオの追加

```python
# scenarios/jitter_comp_scenario.py
from .base_scenario import BaseScenario
from config.sim_config import get_simulator_config
import mosaik

class JitterCompScenario(BaseScenario):
    @property
    def scenario_name(self) -> str:
        return "JitterComp"

    @property
    def scenario_description(self) -> str:
        return "HILS with Jitter Compensation"

    def create_world(self) -> mosaik.World:
        # 実装...
        pass

    # その他のメソッド実装...
```

```python
# main_v2.py に追加
from scenarios import JitterCompScenario

scenario_map = {
    "hils": HILSScenario,
    "rt": RTScenario,
    "inverse_comp": InverseCompScenario,
    "pure_python": PurePythonScenario,
    "jitter_comp": JitterCompScenario,  # 追加
}
```

## パラメータ管理

### 環境変数（.env）

パラメータは全て環境変数から読み込み可能:

```bash
# .env ファイル例
SIMULATION_TIME=2.0
TIME_RESOLUTION=0.0001
CMD_DELAY=20
SENSE_DELAY=30
KP=15.0
KD=5.0
TARGET_POSITION=5.0
ENABLE_INVERSE_COMP=True
INVERSE_COMP_GAIN=15.0
```

### コード内での設定

```python
from config.parameters import SimulationParameters, ControlParams

# カスタムパラメータ
params = SimulationParameters(
    simulation_time=5.0,
    time_resolution=0.0001,
    control=ControlParams(
        kp=20.0,
        kd=10.0,
        target_position=10.0,
    ),
)

scenario = HILSScenario(params)
scenario.run()
```

## 後方互換性

既存の`main_*.py`ファイルは保持されているため、v1のスクリプトは引き続き動作します:

```bash
# v1（従来通り動作）
python main_hils.py
python main_hils_rt.py
python main_hils_with_inverse_comp.py
python main_pure_python.py

# v2（新しい方法）
python main_v2.py hils
python main_v2.py rt
python main_v2.py inverse_comp
python main_v2.py pure_python
```

## まとめ

v2アーキテクチャの主な利点:

1. **コードの再利用性**: 共通ロジックを基底クラスに集約
2. **保守性の向上**: パラメータ管理が一元化
3. **拡張性**: 新しいシナリオの追加が容易
4. **テスト容易性**: 各コンポーネントを独立してテスト可能
5. **明確な責任分離**: 設定、シナリオ、実行が明確に分離
6. **後方互換性**: 既存のスクリプトも引き続き動作

今後は、新しいシナリオや機能の追加はv2アーキテクチャをベースに行うことを推奨します。
