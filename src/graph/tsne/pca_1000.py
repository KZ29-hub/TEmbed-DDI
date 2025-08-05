import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from matplotlib.lines import Line2D

def readJSONL(fp):
    res = []
    with open(fp, "r", encoding='utf-8') as f:
        for line in f.readlines():
            res.append(json.loads(line))
    return res  

# 数据路径
merged_pool_path = "/root/autodl-tmp/multiview-project-1023/tsne/data/zong_1000.jsonl"
embeddings = np.load( "/root/autodl-tmp/multiview-project-1023/tsne/data/embedding_pool.npy")
merged_pool_json = readJSONL(merged_pool_path)

# 获取标签
labels = [example["source"] for example in merged_pool_json]

# 标签编码
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# 创建 DataFrame 以便于选择样本
df = pd.DataFrame({'embedding': list(embeddings), 'label': encoded_labels})

# 选择每个标签的 1000 个样本
sampled_df = df.groupby('label').apply(lambda x: x.sample(n=1000, random_state=0) if len(x) >= 1000 else x)
sampled_embeddings = np.array(list(sampled_df['embedding']))
sampled_encoded_labels = sampled_df['label'].values
print(sampled_embeddings.shape)

# 计算类别数
num_classes = len(set(sampled_encoded_labels))

# 使用 Seaborn 创建调色板
palette = sns.color_palette("husl", num_classes)

# 使用 PCA 降维到 2D
pca_2d = PCA(n_components=2)
embeddings_2d = pca_2d.fit_transform(sampled_embeddings)

# 绘制 2D 图并保存
plt.figure(figsize=(10, 6))
sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=sampled_encoded_labels, palette=palette, legend=False)
plt.title('PCA 2D Visualization')
plt.xlabel('PCA1')
plt.ylabel('PCA2')

# 创建自定义图例
handles = [Line2D([0], [0], marker='o', color='w', label=label_encoder.inverse_transform([i])[0],
                  markerfacecolor=palette[i], markersize=10) for i in range(num_classes)]
plt.legend(handles=handles, title='Labels')
plt.savefig('/root/autodl-tmp/multiview-project-1023/tsne/graph/pca_2d_1000.png')  # 保存 2D 图像
plt.close()

# 使用 PCA 降维到 3D
pca_3d = PCA(n_components=3)
embeddings_3d = pca_3d.fit_transform(sampled_embeddings)

# 绘制 3D 图并保存
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 使用相同的颜色映射，但直接使用 palette
scatter = ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], 
                     c=[palette[i] for i in sampled_encoded_labels], s=50)

# 创建自定义图例
handles = [Line2D([0], [0], marker='o', color='w', label=label_encoder.inverse_transform([i])[0],
                  markerfacecolor=palette[i], markersize=10) for i in range(num_classes)]
ax.legend(handles=handles, title="Labels")

ax.set_title('PCA 3D Visualization')
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')
plt.savefig('/root/autodl-tmp/multiview-project-1023/tsne/graph/pca_3d_1000.png')  # 保存 3D 图像
plt.close()