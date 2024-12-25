import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load data
file_path = "C:\\Desktop\GA+QSVM\german_credit_4.csv"  # Cập nhật đường dẫn file của bạn


credit_df = pd.read_csv(file_path, index_col=0)

# Lấy ngẫu nhiên 100 mẫu từ dataset
credit_df = credit_df.sample(n=100, random_state=42)

# Xử lý giá trị thiếu
credit_df = credit_df.fillna(value="not available")

# Mã hóa các cột phân loại thành số
credit_df['Sex'] = credit_df['Sex'].map({'male': 1, 'female': 2})
credit_df['Housing'] = credit_df['Housing'].map({'own': 1, 'rent': 2, 'free': 3})
credit_df['Saving accounts'] = credit_df['Saving accounts'].map({
    'not available': 0, 'little': 1, 'moderate': 2, 'quite rich': 3, 'rich': 4
})

# Thêm cột 'Checking account' với giá trị mặc định nếu cần
credit_df['Checking account'] = 'not available'
credit_df['Checking account'] = credit_df['Checking account'].map({
    'not available': 0, 'little': 1, 'moderate': 2, 'quite rich': 3, 'rich': 4
})

# Mã hóa nhãn 'Risk' thành số
credit_df['Risk'] = credit_df['Risk'].map({'good': 0, 'bad': 1})

# Tách nhãn và đặc trưng
X = credit_df.drop(['Risk'], axis=1)
y = credit_df['Risk']

# Chuyển đổi sang mảng numpy
X = X.to_numpy()
y = y.to_numpy()

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

# Chuẩn hóa các đặc trưng
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Kiểm tra kết quả
print(f"Kiểu dữ liệu của X_train: {type(X_train)}, kích thước: {X_train.shape}")
print(f"Kiểu dữ liệu của y_train: {type(y_train)}, kích thước: {y_train.shape}")
