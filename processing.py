import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

def process_and_train():
    # 1. Data Loading
    try:
        df = pd.read_csv('heart.csv')
    except FileNotFoundError:
        return None, "File 'heart.csv' tidak ditemukan.", None, None, None, None

    # 2. Pembagian Data (Metode: Train-Test Split dengan Stratify)
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Cek Missing Values
    missing_info = "Tidak ada missing values yang ditemukan."
    if X_train.isnull().sum().sum() > 0:
        missing_info = "Missing values ditemukan dan ditangani."
        # Karena tidak ada missing values handling tidak perlu dilakukan
        # X_train = X_train.fillna(X_train.median())
        # X_test = X_test.fillna(X_train.median())
    
    # 4. Transformasi Data (Metode: StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    # 5. Imbalanced Data (Metode: SMOTE Oversampling)
    balance_info = f"Distribusi kelas sebelum SMOTE:\n{y_train.value_counts(normalize=True).to_string()}"
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    balance_info += f"\nDistribusi kelas setelah SMOTE:\n{pd.Series(y_train_resampled).value_counts(normalize=True).to_string()}"

    # 6. Seleksi Fitur (Metode: SelectKBest dengan ANOVA F-test)
    k_best = 10
    selector = SelectKBest(score_func=f_classif, k=k_best)
    X_train_selected = selector.fit_transform(X_train_resampled, y_train_resampled)
    X_test_selected = selector.transform(X_test_scaled)
    
    selected_features = X.columns[selector.get_support()]
    feature_info = f"Terpilih {k_best} fitur terbaik: {', '.join(selected_features)}"

    # 7. Algoritma Decision Tree 
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train_selected, y_train_resampled)
    
    y_pred = model.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    reports = (missing_info, balance_info, feature_info)
    
    return model, reports, accuracy, scaler, selector, selected_features