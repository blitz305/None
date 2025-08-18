import pickle


def inspect_pkl_data(file_path, num_samples_to_show=2):
    """
    Loads a .pkl file and prints the structure of the first few samples.
    """
    print("=" * 80)
    print(f"🕵️  Inspecting file: {file_path}")
    print("=" * 80)

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        print(f"✅ File loaded successfully.")
        print(f"➡️ Data type: {type(data)}")

        if isinstance(data, list):
            print(f"➡️ Total number of items in the list: {len(data)}")

            if len(data) == 0:
                print("⚠️ The list is empty. Nothing to inspect.")
                return

            print(f"\n--- Now showing the first {num_samples_to_show} item(s) in detail ---\n")

            for i in range(min(num_samples_to_show, len(data))):
                print(f"--- Item #{i + 1} ---")
                sample = data[i]

                if isinstance(sample, dict):
                    print(f"Item is a dictionary. Here are its keys and the type of their values:")
                    for key, value in sample.items():
                        value_type = type(value)

                        # 为了更清晰，如果是列表，我们看一下列表里装的是什么
                        if isinstance(value, list) and value:
                            inner_type = type(value[0])
                            print(f"  - Key: '{key}',  Value Type: list of {inner_type.__name__}")
                        else:
                            print(f"  - Key: '{key}',  Value Type: {value_type.__name__}")

                    # 打印一些关键字段的实际内容，帮助我们理解
                    print("\n  Let's look at some actual content:")
                    if 'dialogue_id' in sample:
                        print(f"    'dialogue_id': {sample['dialogue_id']}")
                    if 'turn_id' in sample:
                        print(f"    'turn_id': {sample['turn_id']}")
                    if 'question' in sample:
                        print(f"    'question': '{sample['question'][:100]}...'")  # 截断一下避免太长
                    if 'answers' in sample:
                        print(f"    'answers': {sample['answers']}")
                    if 'turns' in sample:
                        print(f"    'turns' is a list with {len(sample['turns'])} items.")
                        if sample['turns']:
                            print(f"      The first turn's keys are: {sample['turns'][0].keys()}")

                else:
                    print(f"Item is of type {type(sample)}. Content: {sample}")

                print("\n" + "-" * 20 + "\n")

        else:
            print(f"⚠️ The data is not a list. Here is a preview: {str(data)[:500]}")

    except FileNotFoundError:
        print(f"❌ ERROR: File not found at '{file_path}'. Please check the path.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")


if __name__ == '__main__':
    # ==================================================================
    # 只需要修改下面这个路径，指向您想检查的 .pkl 文件
    # ==================================================================
    # 请使用您原始的数据文件，而不是您尝试转换的那个
    path_to_inspect = r"D:\project\REANO\data\train_with_relevant_triples_wounkrel.pkl"

    inspect_pkl_data(path_to_inspect, num_samples_to_show=2)