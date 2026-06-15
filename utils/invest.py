import pickle
import numpy as np


# 캡틴의 함수를 로보코 센세 스타일로 마개조!
def show_dict(data, name="root", indent=0):
    # 들여쓰기 공백 (4칸씩 띄우기!)
    pad = " " * (indent * 4)
    
    # 1. 📦 딕셔너리일 때 (이름 먼저 출력하고 내용물 파헤치기!)
    if isinstance(data, dict):
        print(f"{pad}📦 {name} (dict, keys: {len(data)})")
        for k, v in data.items():
            show_dict(v, name=k, indent=indent + 1)
            
    # 2. 📜 리스트나 튜플일 때 (캡틴의 아이디어대로 터미널 폭발 방지!)
    elif isinstance(data, (list, tuple)):
        print(f"{pad}📜 {name} ({type(data).__name__}, len: {len(data)})")
        if len(data) > 0:
            show_dict(data[0], name="[0] (sample)", indent=indent + 1)
            
    # 3. 🧊 Numpy 배열일 때 (차원과 타입 출력)
    elif isinstance(data, np.ndarray):
        print(f"{pad}🧊 {name}: ndarray, shape={data.shape}, dtype={data.dtype}")
        
    # 4. 🔹 기본 자료형 (숫자, 문자열, 불리언 등)
    elif isinstance(data, (int, float, str, bool, np.generic)):
        # 문자열이 너무 길면 터미널 더러워지니까 50자에서 컷!
        val_str = str(data)
        if len(val_str) > 50:
            val_str = val_str[:47] + "..."
        print(f"{pad}🔹 {name}: {val_str} ({type(data).__name__})")
        
    # 5. ❓ 그 외의 낯선 객체들
    else:
        print(f"{pad}❓ {name}: {type(data).__name__}")


pkl_path = 'nuscenes_dbinfos_train.pkl'

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)


show_dict(data, name=pkl_path)