# test_library.py
from core.object_library import get_objects_for_room, get_object_template

if __name__ == "__main__":
    print("\n>>> 测试 1: 获取厨房物体 (应该包含 kitchen.yaml + common.yaml)")
    kitchen_objs = get_objects_for_room("kitchen")
    print(f"厨房物体: {kitchen_objs}")
    
    print("\n>>> 测试 2: 获取单个物体属性")
    # 假设你在 kitchen.yaml 里定义了 fridge
    if "fridge" in kitchen_objs:
        tmpl = get_object_template("fridge")
        print(f"冰箱属性: {tmpl}")
    else:
        print("❌ 没找到 fridge，请检查 kitchen.yaml 是否写了 fridge")

    print("\n>>> 测试 3: 获取卧室物体")
    bedroom_objs = get_objects_for_room("bedroom")
    print(f"卧室物体: {bedroom_objs}")