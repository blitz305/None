import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# 假设 mem2n.py 和这个脚本在同一个文件夹下
from mem2n import MemN2N


# --- 1. 自动生成一个用于测试的数据集 ---
# 这个数据集模仿了 bAbI 任务，专门用于测试记忆网络

def create_babi_like_dataset():
    """创建一个包含记忆推理任务的微型数据集"""
    dataset = []

    # 模板1: 简单的位置查找
    names = ["张三", "李四", "王五"]
    locations = ["厨房", "花园", "卧室"]
    for name in names:
        for loc in locations:
            story = [f"{n} 在 {l}。" for n, l in zip(names, locations) if n != name]
            story.append(f"{name} 在 {loc}。")
            random.shuffle(story)  # 打乱事实顺序
            query = f"{name} 在哪里？"
            answer = loc
            dataset.append((story, query, answer))

    # 模板2: 位置更新 (测试记忆更新能力)
    name = "赵六"
    loc1, loc2 = "客厅", "书房"
    story = [
        f"{name} 在 {loc1}。",
        "电话响了。",
        f"{name} 走到了 {loc2}。"
    ]
    query = f"{name} 在哪里？"
    answer = loc2
    dataset.append((story, query, answer))

    return dataset


# --- 2. 定义一个测试“外壳”模型 ---
# 这个模型使用 MemN2N 的输出进行最终的分类预测

class MemN2N_TestHarness(nn.Module):
    def __init__(self, memn2n_module, vocab_size):
        super(MemN2N_TestHarness, self).__init__()
        self.memn2n = memn2n_module
        # 添加一个线性层，将记忆网络的输出映射到整个词汇表，以进行预测
        self.output_layer = nn.Linear(memn2n_module.embedding_size, vocab_size)

    def forward(self, story, query):
        # 得到 MemN2N 对故事和问题的理解向量
        memory_vector = self.memn2n(story, query)
        # 预测词汇表中每个词作为答案的得分
        scores = self.output_layer(memory_vector)
        return scores


# --- 3. 主训练和评估流程 ---

if __name__ == '__main__':
    # --- 设置超参数 ---
    MEMORY_SIZE = 10  # 故事最多包含多少句话
    SENTENCE_SIZE = 8  # 每句话最多包含多少个词
    EMBEDDING_SIZE = 64
    LEARNING_RATE = 0.01
    EPOCHS = 50

    # --- 准备数据和词汇表 ---
    print("1. Creating a synthetic bAbI-like dataset...")
    dataset = create_babi_like_dataset()

    vocab = set()
    for story, query, answer in dataset:
        for sentence in story:
            vocab.update(sentence.replace("？", " ？").replace("。", " 。").split())
        vocab.update(query.replace("？", " ？").replace("。", " 。").split())
        vocab.add(answer)

    vocab.add("<PAD>")  # 添加填充符
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    ix_to_word = {i: word for i, word in enumerate(vocab)}
    VOCAB_SIZE = len(vocab)

    print(f"   Vocabulary size: {VOCAB_SIZE}")
    print(f"   Dataset size: {len(dataset)}")

    # --- 实例化模型、损失函数和优化器 ---
    print("\n2. Initializing the MemN2N module and test harness...")
    memn2n_instance = MemN2N(
        vocab_size=VOCAB_SIZE,
        embedding_size=EMBEDDING_SIZE,
        memory_size=MEMORY_SIZE,
        sentence_size=SENTENCE_SIZE,
        hops=3
    )
    model = MemN2N_TestHarness(memn2n_instance, VOCAB_SIZE)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


    # --- 辅助函数：将文本转换为张量 ---
    def prepare_data(story, query, answer_word):
        # 故事处理
        story_indices = []
        for sentence in story:
            words = sentence.replace("？", " ？").replace("。", " 。").split()
            sent_ix = [word_to_ix[w] for w in words]
            # 填充或截断句子
            sent_ix += [word_to_ix["<PAD>"]] * (SENTENCE_SIZE - len(sent_ix))
            story_indices.append(sent_ix[:SENTENCE_SIZE])

        # 填充或截断故事
        empty_sentence = [word_to_ix["<PAD>"]] * SENTENCE_SIZE
        story_indices += [empty_sentence] * (MEMORY_SIZE - len(story_indices))
        story_tensor = torch.tensor(story_indices[:MEMORY_SIZE], dtype=torch.long)

        # 问题处理
        query_words = query.replace("？", " ？").replace("。", " 。").split()
        query_ix = [word_to_ix[w] for w in query_words]
        query_ix += [word_to_ix["<PAD>"]] * (SENTENCE_SIZE - len(query_ix))
        query_tensor = torch.tensor(query_ix[:SENTENCE_SIZE], dtype=torch.long)

        # 答案处理
        answer_tensor = torch.tensor([word_to_ix[answer_word]], dtype=torch.long)

        # 增加 batch 维度
        return story_tensor.unsqueeze(0), query_tensor.unsqueeze(0), answer_tensor


    # --- 开始训练 ---
    print("\n3. Starting training loop...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for story, query, answer in dataset:
            model.zero_grad()

            story_tensor, query_tensor, answer_tensor = prepare_data(story, query, answer)

            scores = model(story_tensor, query_tensor)

            loss = loss_function(scores, answer_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch + 1}/{EPOCHS}, Average Loss: {total_loss / len(dataset):.4f}")

    print("\n4. Training finished. Evaluating performance...")
    with torch.no_grad():
        correct_count = 0
        for story, query, answer in dataset:
            story_tensor, query_tensor, _ = prepare_data(story, query, answer)
            scores = model(story_tensor, query_tensor)

            # 找到得分最高的词的索引
            _, predicted_ix = torch.max(scores, 1)
            predicted_word = ix_to_word[predicted_ix.item()]

            if predicted_word == answer:
                correct_count += 1

            print(f"   Story: {story}")
            print(f"   Query: {query}")
            print(f"   Expected Answer: '{answer}', Model Prediction: '{predicted_word}'")
            print("-" * 20)

        print(f"\n✅ Final Accuracy: {correct_count / len(dataset) * 100:.2f}%")