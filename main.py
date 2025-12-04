import numpy as np
import matplotlib.pyplot as plt
import json, os

# 執行後產生名為"output"的資料夾
if not os.path.exists("output"):
    os.makedirs("output")

# ----- 載入與預處理資料-----
# 使用loadtxt載入資料，以逗號分隔，型態是np.float32，跳過標籤row1
file = np.loadtxt("Group_A_train.csv", delimiter=",", dtype=np.float32, skiprows=1) 
y = file[:, 0].astype(np.int64) # 取出label(column 0)
X = file[:, 1:] / 255.0  # 取出x像素值(column 1~最後)，並將資料正規化：除以255.0，將範圍縮到[0,1]

# 打散資料
np.random.seed(42)  # 設定np的亂數種子
index = np.arange(len(X)) # 產生 0 ~ N-1 的陣列(N = len(X))當作row index
np.random.shuffle(index) # 打亂陣列排序，例如原本是[0,1,2,3,4,5]打亂變成[1,3,5,0,4,2]
X = X[index] # 將打亂後的index重新排序 X 的順序
y = y[index] # 將打亂後的index重新排序 y 的順序

# 分成 80% train set 和 20% validation set
total_len = len(X) # 計算資料X長度總共多少
split = int(total_len * 0.8) # 找出80%的位置
X_train = X[:split] # 訓練集[從頭開始~80%)，含頭不含尾
y_train = y[:split] 
X_val = X[split:] # 驗證集從80%開始~結束
y_val = y[split:]

# -----One-Hot Encoding-----
# 把整數標籤轉成one-hot向量
def one_hot_encode(labels, num_classes):    #定義函式，參數為labels代表所有標籤(一維Array)，num_classes代表類別的總數
    n = len(labels) # 拿到labels的長度
    one_hot = np.zeros((n, num_classes)) # 建一個 n * num_classes 大小的矩陣，內容都先放0

    # numpy中進階索引用法，兩個等長的整數索引矩陣(n,)
    # 放進雙陣列索引one_hot[row_idx_arr, col_idx_arr]，會逐一配對成[(0, labels[0]), (1,labels[1]),...,(n-1, labels[n-1])]
    one_hot[np.arange(n), labels] = 1 # np.arange(n)會產生 0 ~ n-1 的陣列，並在[(0, labels[0]), (1,labels[1]),...,(n-1, labels[n-1])]各別位置上寫入1
    return one_hot  # 回傳矩陣

# 取得類別數量
unique_labels = np.unique(y)  # 找出總共有幾個不重複label
num_classes = len(unique_labels) # label的長度代表有幾個class

# 建立標籤映射（將原始標籤映射到 0, 1, 2, 3），作為字典來用
# enumerate(unique_labels)把unique_labels(假設是(0,1,8,9))轉成(0,0),(1,1),(2,8),(3,9)
# for idx, label 將 enumerate(unique_labels)產生的配對拆成兩個變數idx和label
# label: idx for 則是把key設成label，value設成idx，最後得到label_to_idx={0:0, 1:1, 8:2, 9:3}
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
#idx_to_label剛好與label_to_idx相反，得到idx_to_label={0:0, 1:1, 2:8, 3:9}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

# 轉換標籤，把原始標籤用剛才建好的字典去轉換成one-hot
# 拿y_train裡的label對應label_to_idx裡面的label轉換，再用np.array將list轉成numpy陣列型態
y_train_mapped = np.array([label_to_idx[label] for label in y_train]) # shape=(y_train,)
# y_val也是一樣
y_val_mapped = np.array([label_to_idx[label] for label in y_val]) # shape=(y_val,)

y_train_onehot = one_hot_encode(y_train_mapped, num_classes) # 套到one_hot_encode函式，得到每row只有一個位置會是1，其他是0
y_val_onehot = one_hot_encode(y_val_mapped, num_classes)

# -----定義函數-----
def softmax(z):  # z.shape=(N筆資料, K種類別)
    z1 = z - np.max(z, axis=1, keepdims=True) # 將原本的z先減掉每row的最大值，(亦即(ni) = (ni)-max(ni))，同時保持維度相同
    exp_z = np.exp(z1) # 取出 e^(ni)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True) # 每row的e^(ni)除以該row的總和

def cross_entropy_loss(y_pred, y_true): # loss函式，計算與預測值的落差
    # 公式: -Σ(j=1 to K) yj * log(y_hat(j))
    y_pred_safe = y_pred + 1e-10 # 避免出現log(0) = -INF，因此加上非常小的值讓log裡面永遠大於1
    cross_entropy = -y_true * np.log(y_pred_safe) #套入公式
    loss = np.sum(cross_entropy) / y_pred.shape[0]  #將所有cross_entropy加總後除以總資料筆數得到平均loss
    return loss

def relu(x):
    return np.maximum(0, x)
def d_relu(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1/ (1 + np.exp(-1))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def d_tanh(x):
    return 1 - tanh(x)**2

def predict(X, W, b):   # 用來將預測好的W和b對測試集做預測
    z = X @ W + b       # 得出每row的資料對k個類別的logits
    a = softmax(z)      # 將每row的logits轉成機率
    return np.argmax(a, axis=1) # 取得最大的機率，得到(n,)的整數陣列

def calculate_accuracy(y_pred, y_true):  # 計算準確率
    return np.mean(y_pred == y_true)     # 1/N sigma(1或-1)，預測正確=1，不對=-1

# -----初始化參數-----
input_dim = X_train.shape[1]  # 784 pixel
output_dim = num_classes  # 4，分別是0,1,8,9
hidden_dim_1 = 256
hidden_dim_2 = 128

# 初始化權重和偏差
# (input_dim, output_dim)是(784,4)的矩陣，數字先隨機產生，row對應的是每個pixel，col對應的是每個類別
# 為了避免一開始 W 過大導致softmax變得太極端，因此乘上0.01讓數字變小，再指派給W
W1 = np.random.randn(input_dim, output_dim) * 0.01 
b1 = np.zeros(hidden_dim_1) # 最初設成0，讓資料不偏重任何一類
W2 = np.random.randn(input_dim, output_dim) * 0.01 
b2 = np.zeros(hidden_dim_2) # 最初設成0，讓資料不偏重任何一類
W3 = np.random.randn(input_dim, output_dim) * 0.01 
b3 = np.zeros(output_dim) # 最初設成0，讓資料不偏重任何一類

# -----訓練參數設定-----
learning_rate = 0.001   # 設定 學習率
maxEpochs = 500         # 設定 epoch 
batch_size = 64         # 設定 每個batch的大小 
patience = 15
no_improve = 0
best_val_loss = float('inf')

# 用於記錄訓練過程
train_losses = []       # 紀錄 訓練loss
train_accuracies = []   # 紀錄 訓練準確率
val_losses = []         # 紀錄 驗證loss
val_accuracies = []     # 紀錄 驗證準確率

# -----訓練迴圈-----
for epoch in range(maxEpochs):   # 跑每個epoch，總告跑maxEpoch次
    # mini-batch 訓練
    indices = np.arange(len(X_train))  # 產生0 ~ N-1 的陣列(N = 訓練集樣本數)
    np.random.shuffle(indices) # 對每個樣本順序再隨機打亂一次
    
    epoch_loss = 0  # 記錄所有epoch的batch loss，之後要取平均用
    num_batches = 0 # 記錄這是第幾個batch
    
    for start_idx in range(0, len(X_train), batch_size):  # 以batch_size將訓練集切成各個mini-batch
        end_idx = min(start_idx + batch_size, len(X_train))  # 防止切batch到最後超出邊界
        batch_indices = indices[start_idx:end_idx]  # 此次epoch的索引範圍
        
        X_batch = X_train[batch_indices]    # 取出此次epoch的X_train
        y_batch = y_train_onehot[batch_indices] # 取出此次epoch的y_train_onehot
        
        # ----- Feedforward -----
        a0 = X_batch
        n1 = W1 @ a0 + b1
        a1 = relu(n1)
        n2 = W2 @ a1 + b2
        a2 = relu(n2)
        n3 = W3 @ a2 + b3
        a3 = softmax(n3)

        # ----- Backward -----
        # Output layer error
        batch_loss_3 = cross_entropy_loss(a3, y_batch) # 此次batch的平均loss
        if(batch_loss_3 < best_val_loss):
            batch_loss_3 = best_val_loss
            no_improve = 0
        else:
            no_improve += 1

        if(no_improve >= patience):
            break

        epoch_loss += batch_loss_3    # 累積到epoch的總loss量
        num_batches += 1    # 批次數+1
        
        # Hidden layer errors
        batch_loss_3 = np.dot((W4.T @ batch_loss_4), d_relu(n3))
        batch_loss_2 = np.dot((W3.T @ batch_loss_3), d_relu(n2))
        batch_loss_1 = np.dot((W2.T @ batch_loss_2), d_relu(n1))

        # # 梯度計算
        # dz = y_batch - a3  # 誤差 = y - y_hat，shape: (batch_size, num_classes)
        # # 將X轉置與dz做矩陣乘法，再除以batch長度
        # # shape = (input_dim, batch_size) @ (batch_size, num_classes) = (input_dim, num_classes)
        # dW = X_batch.T @ dz / len(X_batch) 
        # db = np.mean(dz, axis=0)  # 1/B sigma(dz)，shape: (num_classes,)
        
        # Update weights and biases 更新參數
        W3 = W3 + learning_rate * batch_loss_3 * a2.T
        b3 = b3 + learning_rate * a3.T
        W2 = W2 + learning_rate * batch_loss_2 * a1.T
        b2 = b2 + learning_rate * a2.T
        W1 = W1 + learning_rate * batch_loss_1 * a0.T
        b1 = b1 + learning_rate * a1.T

    # 跑完所有batch
    # 計算平均損失
    avg_train_loss = epoch_loss / num_batches
    
    # 計算訓練集準確率
    train_pred = predict(X_train, W, b)  # 預測訓練集
    train_acc = calculate_accuracy(train_pred, y_train_mapped)  # 與答案比對，計算準確率
    
    # 計算驗證集loss和準確率
    z_val = X_val @ W + b   # 取得logits
    a_val = softmax(z_val)  # 計算機率
    val_loss = cross_entropy_loss(a_val, y_val_onehot)  # 計算loss
    val_pred = predict(X_val, W, b) # 預測類別
    val_acc = calculate_accuracy(val_pred, y_val_mapped) # 計算準確率
    
    # 記錄結果
    train_losses.append(avg_train_loss) # 紀錄 訓練集平均loss
    train_accuracies.append(train_acc)  # 紀錄 訓練集準確率
    val_losses.append(val_loss)         # 紀錄 驗證集loss
    val_accuracies.append(val_acc)      # 紀錄 驗證集準確率

final_epoch = len(train_losses) # 沒有寫early stopping，因此這裡final_epoch就是maxEpoch

# -----輸出 output.json -----
with open('output/output.json', 'w') as file:   # 檔案寫入
    json.dump({
        "Learning rate": learning_rate,
        "Epoch": final_epoch,
        "Batch size": batch_size,
        "Final train accuracy": round(train_accuracies[-1], 6), # 取出最終訓練集準確率的值，小數點保留6位
        "Validation accuracy": round(val_accuracies[-1], 6), # 取出最終驗證集準確率的值，小數點保留6位
        "Final train loss": round(train_losses[-1], 6), # 取出最終訓練集loss的值，小數點保留6位
        "Final validation loss": round(val_losses[-1], 6) # 取出最終訓練集loss的值，小數點保留6位
        }, file, indent=4) # indent=4 每一層縮排4個空格

# ----- 繪製學習曲線 -----
# Accuracy圖
plt.figure(figsize=(6, 4)) # 設定圖片⼤⼩
epochs_range = range(1, final_epoch + 1) # 建立x軸要化的epoch範圍，從1到最後的epoch，不含右邊所以要加上1
plt.plot(epochs_range, train_accuracies, color="Blue", label='Train accuracy') # 繪製資料：參數為x,y,color,label
plt.plot(epochs_range, val_accuracies, color='Orange', label='Validation accuracy') # 繪製資料
plt.xlabel('Epoch') # 設定x軸文字
plt.ylabel('Accuracy') # 設定y軸文字
plt.title('GroupB_Accuracy') # 圖片標題
plt.legend(loc="lower right") # label 顯示位置
plt.savefig('output/output_accuracy.png') # 儲存圖片

# Loss圖
plt.figure(figsize=(6, 4)) # 設定圖片⼤⼩
plt.plot(epochs_range, train_losses, color='Blue', label='Train loss') # 繪製資料
plt.plot(epochs_range, val_losses, color='Orange', label='Validation loss') # 繪製資料
plt.xlabel('Epoch') # 設定x軸文字
plt.ylabel('Loss') # 設定y軸文字
plt.title('GroupB_Loss') # 圖片標題
plt.legend(loc="upper right") # label 顯示位置
plt.savefig('output/output_loss.png') # 儲存圖片
plt.show() # 顯示圖片
plt.clf()  # 清除圖片

# -----載入測試資料並預測-----
# 使用loadtxt載入資料，以逗號分隔，型態是np.float32，跳過標籤row1
test_file = np.loadtxt("Group_B_test.csv", delimiter=",", dtype=np.float32, skiprows=1) 
# 取出x像素值(column 1~最後)，並將資料正規化：除以255.0，將範圍縮到[0,1]
X_test = test_file / 255.0 

# 預測 測試集資料
test_pred_mapped = predict(X_test, W, b)

# 將預測結果對應到原本的標籤
test_pred = [int(idx_to_label[idx]) for idx in test_pred_mapped]

# ----- 儲存預測結果-----
with open('output/test_set_prediction.json', 'w') as file: # 檔案寫入
    json.dump({
        "Predictions": test_pred
        }, file, indent=4) # indent=4 每一層縮排4個空格
