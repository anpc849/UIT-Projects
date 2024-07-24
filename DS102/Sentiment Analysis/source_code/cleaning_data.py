from train_model import clean_text
import re
import multiprocessing as mp

data = pd.read_csv("labeled_data.csv")[['Comment', 'sentiment']]
data.drop_duplicates(subset=['Comment'], inplace=True)

# Loại bỏ các thuộc tính mặc định của shopee
# "Chất liệu:", "Màu sắc:", "Đúng với mô tả:"
lst_remove = ['Chất liệu:', 'Màu sắc:', 'Đúng với mô tả:']
pattern = '|'.join(map(re.escape,lst_remove))
data.Comment = data.Comment.str.replace(pattern, "")


data['Comment'] = data.Comment.str.replace('[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]','')
p = mp.Pool(mp.cpu_count()) # Data parallelism Object
data['Comment'] = p.map(clean_text, data['Comment'])
data = data.dropna()