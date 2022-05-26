from data import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
import lightgbm as lgb

# prepare data
train = pd.merge(train_info, y_train, on='id_bh', how="inner")
train = pd.merge(train, train_work, on='id_bh', how="inner")
train = train.drop_duplicates(subset=['id_bh'], keep='first')

test = pd.merge(test_info, test_work, on='id_bh', how="inner")
test = test.drop_duplicates(subset=['id_bh'], keep='first')

features_full = train.columns.tolist()

features_categorical = ['id_office', 'company_type', 'job/role', 'employee_lv', 'work_days', 'delays',
                        "bithYear", 'gender', 'address_x', 'age', 'id_management', 'address_y']

encoder_cols = ['gender', 'address_x', 'id_management', 'id_office',  'job/role', 'address_y',  'company_type']

for c in ["label", "id_bh", "id", 'from_date', 'to_date']:
    features_full.remove(c)

train_data = train[features_full]
test_id = test['id_bh'].tolist()
test_data = test[features_full]
train_label = train['label']

# Encoder
train_data[encoder_cols] = train_data[encoder_cols].apply(LabelEncoder().fit_transform)
test_data[encoder_cols] = test_data[encoder_cols].apply(LabelEncoder().fit_transform)

# train-val split
X_train, X_valid, y_train, y_valid = train_test_split(train_data, train_label, test_size=0.1, random_state=42)

# model
model = lgb.LGBMClassifier(learning_rate=0.09, max_depth=12, n_estimators=500, random_state=42, subsample=0.8658)

if __name__ == "__main__":
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid), (X_train, y_train)], eval_metric='logloss')

    print('Training accuracy {:.4f}'.format(model.score(X_train, y_train)))
    print('Testing accuracy {:.4f}'.format(model.score(X_valid, y_valid)))

    test_label = model.predict(test_data).astype('int64')
    test["label"] = test_label

    sub = sub.merge(test, on="id_bh")
    sub = sub[["id_bh", "label"]]
    sub.to_csv("./submission.csv", index=False)
