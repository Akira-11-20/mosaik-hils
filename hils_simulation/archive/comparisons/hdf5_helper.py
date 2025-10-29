"""
HDF5データ読み込みヘルパー

新旧両方のHDF5形式に対応したデータ読み込みユーティリティ
"""

import h5py


def load_hdf5_data(h5_path):
    """
    HDF5ファイルからデータを読み込む（階層構造対応）

    Args:
        h5_path: HDF5ファイルのパス

    Returns:
        dict: フラット化されたデータ辞書
            - 旧形式: そのままのキー名
            - 新形式: attr_name_group_name 形式に変換
    """
    hdf5_data = {}
    with h5py.File(h5_path, "r") as f:
        # 旧形式（data/以下にフラット）の対応
        if "data" in f:
            for key in f["data"].keys():
                hdf5_data[key] = f["data"][key][:]
        else:
            # 新形式（ノードごとにグループ化）の対応
            def read_group(group):
                """再帰的にグループを読み込む"""
                for key in group.keys():
                    item = group[key]
                    if isinstance(item, h5py.Group):
                        # グループの場合、再帰的に読み込む
                        read_group(item)
                    elif isinstance(item, h5py.Dataset):
                        # 階層を逆にしてフラット化: group_name/attr -> attr_group_name
                        parts = item.name.split("/")
                        if len(parts) >= 2:
                            # /group_name/attr_name -> attr_name_group_name
                            group_name = parts[1]
                            attr_name = parts[-1]
                            # timeグループは特別扱い（time_msとtime_sはそのまま）
                            if group_name == "time":
                                flat_key = attr_name
                            else:
                                # グループ名のドット(.)をアンダースコアに変換（HDF5グループ名として保存済み）
                                flat_key = f"{attr_name}_{group_name}"
                            hdf5_data[flat_key] = item[:]

            read_group(f)
    return hdf5_data


def get_dataset(hdf5_data, key_suffix):
    """
    指定されたサフィックスを持つキーのデータを取得

    Args:
        hdf5_data: load_hdf5_data()で読み込んだデータ辞書
        key_suffix: キーのサフィックス（例: "position_EnvSim-0.Spacecraft1DOF_0"）

    Returns:
        numpy.ndarray: データ配列、見つからない場合はNone
    """
    # 完全一致を試す
    if key_suffix in hdf5_data:
        return hdf5_data[key_suffix]

    # サフィックスマッチング
    for key in hdf5_data.keys():
        if key.endswith(key_suffix):
            return hdf5_data[key]

    # 見つからない
    return None


def list_available_keys(h5_path):
    """
    HDF5ファイル内の全データセットキーをリスト表示

    Args:
        h5_path: HDF5ファイルのパス
    """
    data = load_hdf5_data(h5_path)
    print(f"\nAvailable keys in {h5_path}:")
    for i, key in enumerate(sorted(data.keys()), 1):
        print(f"  {i:2d}. {key}")
    print(f"\nTotal: {len(data)} datasets")
