from preprocessing import *

train_work = pd.read_csv('data/work_train.csv')
test_work = pd.read_csv('data/work_test.csv')
train_info = pd.read_csv('data/info_train.csv')
test_info = pd.read_csv('data/info_test.csv')
y_train = pd.read_csv('data/label_train.csv')
sub = pd.read_csv("data/label_test.csv")

# replace nan
train_work["id_management"] = train_work["id_management"].astype(str)
train_work["company_type"] = train_work["company_type"].replace(-1, 9)
train_work["company_type"] = train_work["company_type"].astype(str)
train_work["job/role"] = train_work["job/role"].replace(np.nan, "thieu")
train_work["address"] = train_work["address"].replace(np.nan, "viet nam")
train_work["id_office"] = train_work["id_office"].replace(np.nan, "ZZ000ZZ")

train_info["address"] = train_info["address"].replace(np.nan, "viet nam")

test_work["id_management"] = test_work["id_management"].astype(str)
test_work["company_type"] = test_work["company_type"].replace(-1, 9)
test_work["company_type"] = test_work["company_type"].astype(str)
test_work["job/role"] = test_work["job/role"].replace(np.nan, "thieu")
test_work["address"] = test_work["address"].replace(np.nan, "viet nam")
test_work["id_office"] = test_work["id_office"].replace(np.nan, "ZZ000ZZ")

test_info["address"] = test_info["address"].replace(np.nan, "viet nam")

# Loại bỏ tiếng Việt có dấu
train_info["address"] = train_info["address"].apply(no_accent_vietnamese)
train_work["address"] = train_work["address"].apply(no_accent_vietnamese)
train_work["job/role"] = train_work["job/role"].apply(no_accent_vietnamese)

test_info["address"] = test_info["address"].apply(no_accent_vietnamese)
test_work["address"] = test_work["address"].apply(no_accent_vietnamese)
test_work["job/role"] = test_work["job/role"].apply(no_accent_vietnamese)

# Chuẩn hóa xâu thành chữ thường
train_work['job/role'] = train_work['job/role'].str.lower()
test_work['job/role'] = test_work['job/role'].str.lower()

train_work['address'] = train_work['address'].str.lower()
test_work['address'] = test_work['address'].str.lower()

train_info['address'] = train_info['address'].str.lower()
test_info['address'] = test_info['address'].str.lower()

train_info["address"] = train_info["address"].apply(str_normalize)
test_info["address"] = test_info["address"].apply(str_normalize)
train_work["address"] = train_work["address"].apply(str_normalize)
test_work["address"] = test_work["address"].apply(str_normalize)

train_work["job/role"] = train_work["job/role"].apply(str_normalize)
test_work["job/role"] = test_work["job/role"].apply(str_normalize)

# Thêm cột 'age' trong bảng thông tin cá nhân
train_info['age'] = 2022 - train_info['bithYear']
test_info['age'] = 2022 - test_info['bithYear']

# Chuẩn hóa ngày tháng năm
train_work["from_date"] = train_work["from_date"].apply(date_preprocess)
test_work["from_date"] = test_work["from_date"].apply(date_preprocess)

train_work["to_date"] = train_work["to_date"].apply(date_preprocess)
test_work["to_date"] = test_work["to_date"].apply(date_preprocess)

train_work['from_date'] = pd.to_datetime(train_work['from_date'], format='%Y%m%d')
train_work['to_date'] = pd.to_datetime(train_work['to_date'], format='%Y%m%d')

test_work['from_date'] = pd.to_datetime(test_work['from_date'], format='%Y%m%d')
test_work['to_date'] = pd.to_datetime(test_work['to_date'], format='%Y%m%d')

# Thêm cột work_days trong bảng mô tả công việc
train_work['work_days'] = train_work['to_date'] - train_work['from_date']
train_work['work_days'] = train_work['work_days'].dt.days

test_work['work_days'] = test_work['to_date'] - test_work['from_date']
test_work['work_days'] = test_work['work_days'].dt.days

# Thêm cột 'delays' trong bảng mô tả công việc
train_work['delays'] = pd.to_datetime("20220501", format='%Y%m%d') - train_work['to_date']
train_work['delays'] = train_work['delays'].dt.days

test_work['delays'] = pd.to_datetime("20220501", format='%Y%m%d') - test_work['to_date']
test_work['delays'] = test_work['delays'].dt.days

# Nhóm dữ liệu theo id cá nhân trong bảng mô tả công việc
train_work["work_days"] = train_work.groupby(["id_bh"])["work_days"].agg('sum').reindex(train_work["id_bh"].values).values
train_work["employee_lv"] = train_work.groupby(["id_bh"])["employee_lv"].agg('mean').reindex(train_work["id_bh"].values).values
train_work["delays"] = train_work.groupby(["id_bh"])["delays"].agg('min').reindex(train_work["id_bh"].values).values

test_work["work_days"] = test_work.groupby(["id_bh"])["work_days"].agg('sum').reindex(test_work["id_bh"].values).values
test_work["employee_lv"] = test_work.groupby(["id_bh"])["employee_lv"].agg('mean').reindex(test_work["id_bh"].values).values
test_work["delays"] = test_work.groupby(["id_bh"])["delays"].agg('min').reindex(test_work["id_bh"].values).values

