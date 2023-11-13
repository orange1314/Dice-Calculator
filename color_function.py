#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D  # 引入三維繪圖庫

import numpy as np
import pandas as pd
import cv2 
import random

from sklearn.cluster import KMeans,DBSCAN
from sklearn.datasets import make_blobs

import webcolors
import os

import warnings


# In[2]:


os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cluster._kmeans")


# ## RGB 轉換成顏色名稱函數介紹
# 
# ### 功能
# 該函數的主要功能是將給定的 RGB 顏色值轉換為對應的顏色名稱。
# 
# ### 計算方法
# 1. 函數使用 CSS3_HEX_TO_NAMES 字典，其中包含了各種十六進制顏色碼對應的顏色名稱。
# 2. 對於每個十六進制顏色碼，將其轉換為 RGB 顏色值。
# 3. 計算給定 RGB 值與每個 CSS3 顏色的歐式距離的平方。
# 4. 找到距離最小的顏色，將其視為最接近的顏色名稱。
# 
# ### 參數
# - `rgb`: 一個包含三個元素的列表，表示待轉換的 RGB 顏色值。
# 
# ### 返回值
# 返回與給定 RGB 顏色值最接近的顏色名稱。
# 
# ### 注意事項
# - 該函數適用於在 CSS3 規範中定義的顏色名稱。
# - 此方法基於歐式距離的計算，可能不考慮人類對顏色相似性的主觀感受。
# 

# In[3]:


# 顏色名稱函數
def rgb_to_color_name(rgb):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - rgb[0]) ** 2
        gd = (g_c - rgb[1]) ** 2
        bd = (b_c - rgb[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]


# ## Dice 係數介紹
# 
# ### 定義
# Dice 係數是一種用於衡量兩個集合相似度的統計指標。它通常應用於二進制分割問題，例如圖像分割或區域分割。
# 
# ### 計算方法
# Dice 係數的計算公式為：
# 
# $ Dice(A, B) = \frac{2 \times |A \cap B|}{|A| + |B|} $
# 
# 其中：
# - $( A ) 和 ( B )$ 分別是兩個集合。
# - $( |A \cap B| )$ 表示兩個集合的交集的元素個數。
# - $( |A| ) 和 ( |B| )$ 分別表示兩個集合的元素個數。
# 
# ### 特性
# - Dice 係數的取值範圍為 0 到 1，其中 0 表示兩個集合無交集，1 表示兩個集合完全相同。
# - 這個指標對於應對不平衡的數據集（即兩個集合中元素數量差異較大）效果較好。
# - Dice 係數越高，表示兩個集合的相似度越大。
# 
# ### 應用場景
# Dice 係數廣泛應用於醫學影像分割、物體檢測、文本分割等領域，用於評估模型預測結果與實際標籤之間的相似度和準確性。
# 

# ## dice_coeff 函數詳細介紹
# 
# ### 功能
# 這個函數計算兩個二進制矩陣之間的 Dice 係數。Dice 係數用於衡量兩個集合的相似度，通常用於二進制分割問題中。
# 
# ### 參數
# - `matrix_1`: 一個二進制矩陣。
# - `matrix_2`: 另一個二進制矩陣。
# 
# ### 返回值
# 一個浮點數，表示兩個矩陣的 Dice 係數。
# 
# ### 函數實現
# 
# 1. **數據整形**
#    - `matrix_1 = matrix_1.reshape(1, -1)[0]`: 通過 reshape 函數將矩陣壓平成一維數組。
#    - `matrix_2 = matrix_2.reshape(1, -1)[0]`: 同樣壓平第二個矩陣。
# 
# 2. **計算交集**
#    - `intersection = (matrix_1 * matrix_2).sum()`: 通過將兩個矩陣進行逐元素乘法，然後求和，計算二進制矩陣的交集。
# 
# 3. **計算 Dice 係數**
#    - 返回 `(2. * intersection) / (matrix_1.sum() + matrix_2.sum() + 0.000000000000000000001)`: 計算 Dice 係數，避免分母為零。
# 

# In[4]:


def dice_coeff(matrix_1, matrix_2):
    matrix_1 = matrix_1.reshape(1,-1)[0]
    matrix_2 = matrix_2.reshape(1,-1)[0]
    intersection = (matrix_1*matrix_2).sum()
    return (2. * intersection ) / (matrix_1.sum() + matrix_2.sum()+0.000000000000000000001)


# ## color_function 函數詳細介紹
# 
# ### 功能
# 這個函數使用 DBSCAN 和 KMeans 算法進行像素分類和顏色映射，將圖片的每個像素映射到其所屬的顏色類別。
# 
# ### 參數
# - `data`: 二維 NumPy 數組，表示圖像的 RGB 數據。
# - `x`: 圖像的新高度。
# - `y`: 圖像的新寬度。
# 
# ### 返回值
# 一個包含兩個元素的元組：
# 1. 二維 NumPy 數組，表示映射後的顏色矩陣。
# 2. 包含每個聚類中心對應顏色名稱的列表。
# 
# ### 函數實現
# 
# 1. **數據處理**
#    - `data_copy = data.copy()`: 複製原始數據，以避免修改原始數據。
#    - `RGB_data = data_copy.reshape(data_copy.shape[0] * data_copy.shape[1], 3)`: 壓平數據為一維數組。
#    - 抽樣數據以提高效率。
# 
# 2. **DBSCAN 分群**
#    - `dbscan = DBSCAN(eps=0.3, min_samples=5)`: 創建 DBSCAN 分群器。
#    - `cluster_labels = dbscan.fit_predict(RGB_data)`: 使用 DBSCAN 模型進行擬合和預測。
# 
# 3. **KMeans 聚類**
#    - 獲取適合的 K 值。
#    - `kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)`: 創建一個 K-Means 模型。
#    - `kmeans.fit(RGB_data)`: 訓練 K-Means 模型。
# 
# 4. **映射顏色名稱**
#    - `centroids = kmeans.cluster_centers_`: 獲取聚類中心。
#    - `color_name = [rgb_to_color_name(centroids[i] * 255) for i in range(centroids.shape[0])]`: 判斷聚類中心的顏色。
# 
# 5. **建立映射關係**
#    - `color_mapping = {i: color_name[i] for i in range(len(color_name))}`: 將標籤映射到顏色名稱。
# 
# 6. **映射結果轉換**
#    - `mapped_array = np.vectorize(color_mapping.get)(labels)`: 將標籤映射到實際顏色名稱。
#    - 資料 resize 和 reshape。
# 
# 7. **預測像素顏色**
#    - `result = kmeans.predict(data_RGB)`: 利用訓練好的模型預測每個像素的顏色。
# 
# 8. **映射預測的顏色標籤**
#    - `mapped_array = np.vectorize(color_mapping.get)(result)`: 將預測的顏色標籤映射到實際顏色名稱。
# 
# 9. **結果轉換**
#    - `mapped_matrix = mapped_array.reshape(x, y)`: 將結果轉換回矩陣形式。
# 
# 10. **返回結果**
#     - 返回一個包含映射矩陣和顏色名稱的元組。
# 

# In[5]:


def color_function(data, x, y):
    """
    使用 DBSCAN 和 KMeans 算法進行像素分類和顏色映射。

    Parameters:
    - data: 二維 NumPy 數組，表示圖像的 RGB 數據。
    - x: 圖像的新高度。
    - y: 圖像的新寬度。

    Returns:
    - 二維 NumPy 數組，表示映射後的顏色矩陣。
    """
    # 複製原始數據，以避免修改原始數據
    data_copy = data.copy()

    # 將資料壓平為一維數組
    RGB_data = data_copy.reshape(data_copy.shape[0] * data_copy.shape[1], 3)

    # 抽樣數據
    population = list(range(RGB_data.shape[0]))
    sample_size = int(len(population) * 0.001)
    random_sample = random.sample(population, sample_size)
    RGB_data = RGB_data[random_sample]

    # 創建 DBSCAN 分群器
    dbscan = DBSCAN(eps=0.3, min_samples=5)

    # 使用 DBSCAN 模型進行擬合和預測
    cluster_labels = dbscan.fit_predict(RGB_data)

    # 獲取適合的 K 值
    k = len(np.unique(cluster_labels))

    # 創建一個 K-Means 模型並設置 K = k
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)

    # 訓練 K-Means 模型
    kmeans.fit(RGB_data)

    # 獲取聚類中心
    centroids = kmeans.cluster_centers_

    # 獲取每個數據點的聚類標籤
    labels = kmeans.labels_

    # 判斷聚類中心的顏色
    color_name = [rgb_to_color_name(centroids[i] * 255) for i in range(centroids.shape[0])]

    # 將標籤映射到顏色名稱
    color_mapping = {i: color_name[i] for i in range(len(color_name))}
    mapped_array = np.vectorize(color_mapping.get)(labels)

    # 資料resize
    data_copy = cv2.resize(data_copy, (y, x))

    # 資料reshape
    data_RGB = data_copy.reshape(x * y, 3)

    # 利用訓練好的模型預測每個像素的顏色
    result = kmeans.predict(data_RGB)

    # 將預測的顏色標籤映射到實際顏色名稱
    mapped_array = np.vectorize(color_mapping.get)(result)

    # 將結果轉換回矩陣形式
    mapped_matrix = mapped_array.reshape(x, y)

    return (mapped_matrix,color_name)


# ## 輸入資料處理
# 
# - `Data1` 和 `Data2` 是兩張圖片的資料。
# - 透過 `data1, data2 = Data1.copy(), Data2.copy()` 複製兩張圖片的資料，以免修改原始資料。
# 
# ## 確定圖片大小
# 
# - `size1` 和 `size2` 分別表示兩張圖片的大小，是圖片的寬度和高度之和。
# - 根據大小判斷選擇要使用的圖片的寬度 `x` 和高度 `y`。
# 
# ## 獲取顏色分布結果
# 
# - 透過 `color_function` 函數，獲取兩張圖片的顏色分布結果 `result1` 和 `result2`。
# 
# ## 計算顏色集合和交集、聯集、對稱差異
# 
# - 通過 `set` 和集合的操作，計算兩張圖片的顏色集合。
# - 計算交集、聯集和對稱差異的顏色集合。
# 
# ## 顯示迴圈外的圖片
# 
# - 使用 `plt.subplots` 創建一個 1 行 2 列的子圖，顯示兩張圖片。
# - 在子圖中，第一張圖顯示 'Annotations'，第二張圖顯示 'Inference Result'。
# 
# ## 遍歷交集的顏色
# 
# - 使用迴圈遍歷兩張圖片的交集顏色 `intersection_color`。
# - 對每一個交集的顏色，創建兩個零矩陣 `zero_matrix1` 和 `zero_matrix2`。
# 
# ## 計算 Dice 係數
# 
# - 透過 `dice_coeff` 函數計算兩個零矩陣的 Dice 係數。
# 
# ## 顯示零矩陣和 Dice 係數
# 
# - 在新的子圖中，分別顯示 `zero_matrix1` 和 `zero_matrix2`。
# - 在子圖中顯示 Dice 係數和相應的顏色值。
# 
# ## 存儲 Dice 係數和顏色結果
# 
# - 將每個交集的 Dice 係數存儲在列表 `dice` 中。
# - 將 Dice 係數和交集的顏色結果轉換為 DataFrame，並將差異的顏色用 0 填充。
# 
# ## 返回合併的 DataFrame
# 
# - 使用 `pd.concat` 函數將兩個 DataFrame 合併為一個，忽略索引，形成最終的結果。
# - 返回合併後的 DataFrame。

# In[6]:


def Dice_function(Data1, Data2):
    # 複製輸入的資料，以免修改原始資料
    data1, data2 = Data1.copy(), Data2.copy()

    # 確定要使用的圖片大小
    size1 = data1.shape[0] + data1.shape[1]
    size2 = data2.shape[0] + data2.shape[1]

    if size1 > size2:
        x, y = data1.shape[0], data1.shape[1]
    else:
        x, y = data2.shape[0], data2.shape[1]

    # 使用 color_function 函數獲取兩張圖的顏色分布結果
    result1 = color_function(data1, x, y)
    result2 = color_function(data2, x, y)

    # 獲取兩張圖的顏色集合
    set1 = set(result1[1])
    set2 = set(result2[1])

    # 計算顏色的交集、聯集和對稱差異
    intersection_color = list(set1.intersection(set2))
    union_color = list(set1.union(set2))
    symmetric_difference_color = list(set1.symmetric_difference(set2))

    dice = []  # 用來存放 Dice 係數的列表
    fig, axs = plt.subplots(1, 2, figsize=(8, 8))

    # 在迴圈外顯示 Annotations 和 Inference Result 圖片
    axs[0].imshow(data1)
    axs[0].set_title('Annotations')

    axs[1].imshow(data2)
    axs[1].set_title('Inference Result')

    for i in intersection_color:
        zero_matrix1 = np.zeros((x, y))
        zero_matrix2 = np.zeros((x, y))

        # 找出相應顏色在結果中的位置
        non_zero_loc1 = np.where(result1[0] == i)
        non_zero_loc2 = np.where(result2[0] == i)

        # 在 zero_matrix1 和 zero_matrix2 中將相應位置設為 1
        zero_matrix1[non_zero_loc1[0], non_zero_loc1[1]] = 1
        zero_matrix2[non_zero_loc2[0], non_zero_loc2[1]] = 1

        # 計算 Dice 係數
        Dice = dice_coeff(zero_matrix1, zero_matrix2)

        # 創建一個新的子圖，1行2列
        fig, axs = plt.subplots(1, 2, figsize=(8, 8))
        
        # 在子圖中顯示 Zero Matrix 1 和 Zero Matrix 2
        axs[0].imshow(zero_matrix1)
        axs[0].set_title('Annotations')
        
        axs[1].imshow(zero_matrix2)
        axs[1].set_title('Inference Result')

        # 在子圖中顯示 Dice 係數和顏色值
        axs[0].text(0.5, -0.2, f'Dice: {Dice}, Color: {i}', ha='center', va='center', fontsize=12, transform=axs[0].transAxes)

        # 顯示整個圖片
        plt.show()
        dice.append(Dice)

    # 將 Dice 係數和顏色組成 DataFrame
    df1 = pd.DataFrame({'Color': intersection_color, 'Dice Coefficient': dice})
    df2 = pd.DataFrame({'Color': symmetric_difference_color, 'Dice Coefficient': [0] * len(symmetric_difference_color)})
    merged_df = pd.concat([df1, df2], ignore_index=True)

    # merged_df.to_csv("result.csv", encoding='utf-8', index=False)
    return merged_df

