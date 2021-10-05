import unidecode # to remove accents and lower sentences
import random # random choices
import re # regular expression
import pandas as pd
from tqdm import tqdm # to know how much time to finish
import itertools # use to flatten a list of lists
import os # to join path
import numpy as np
tqdm.pandas()


results = dict() # storage

def string_punctuation(text):
    '''
        Remove punctuation
        Input:
            text: str
        Output:
            A string without any punctuation
    '''
    res = re.findall(r'\w+', text)
    return ' '.join(res)

def remove_accent(text):
    '''
        Remove Vietnamese accents
        Input:
            text: str
        Output:
            A string without accents
    '''
    res = unidecode.unidecode(text)
    return string_punctuation(res)

def remove_accent_randomly(text):
    '''
        Randomly remove a few words in a sentence
        Input:
            text: A string
        Output:
            A sentence with a few words having accents removed
    '''
    text_list = np.array(text.split())
    indices = np.random.choice(len(text_list), size=round(0.3*(len(text_list))), replace=False)
    text_list[indices] = [remove_accent(s) for s in text_list[indices]]
    return ' '.join(text_list)

def generate_augmentation(s, num_iters=1000):
    '''
        Create fake data for education. In each iteration, we pick a random case in the dictionary `augmentation`.
        This function will change the dictionary `results`. `results` is a store dictionary
        `results` doesn't contain original data
        Input:
            s: str
            num_iters: The number of iterations (default=1000)
    '''
    global results # storage
    s_no_accents = remove_accent(s).lower()
    results[s] = [s_no_accents]
    
    augmentation = {'th thcs thpt': ['th thcs thpt', 'thpt', 'thptth', 'tkpt', 'tthpt','pt', 'ptth', 'pho thong', 'phổ thông', 'ph0 th0ng', 'pk0 tk0nq',
                        'trung hoc pho thong', 'trung h0.c ph0 th0ng', 'trunq h0c pk0 tk0nq', 
                        'trung học phổ thông', 'truong trung hoc pho thong', 'trường trung học phổ thông', 'tru0`ng trung h0.c ph0 th0ng',
                        'tru0nq trunq h0c pk0 tk0nq', 'truog trug hoc pho thog', 'trg thpt', 'trường thpt', 'truog trug hoc fo thog',
                        'cap 3', 'cấp 3', 'pho thong trung hoc', 'ph0 th0ng trung h0c', 'pk0 tk0nq trunq h0c',
                        'phổ thông trung học', 'fo thog trug hoc','truong pho thong trung hoc', 
                        'trường phổ thông trung học', 'truog fo thog trug hoc', 'tru0ng ph0 th0ng trung h0c', 'tru0nq pk0 tk0nq trunq h0c', 
                        'trường cấp 2-3', 'trường cap 3', 'trường cấp 3',
                        'trường cấp ba'],
                    'thpt dtnt': ['thpt dtnt', 'trung học phổ thông dân tộc nội trú', 'trug hoc fo thog dan toc noj tru' 
                                'trunq h0c pk0 tk0nq dan t0c n0j tru', 'trường thpt dân tộc nội trú', 'trường thpt dan toc noj tru',
                                'trường thpt dan t0c n0j tru'],
                    'thcs thpt': ['thcs thpt', 'thpt', 'thptth', 'tkpt', 'tthpt','pt', 'ptth', 'pho thong', 'phổ thông', 'ph0 th0ng', 'pk0 tk0nq',
                        'trung hoc pho thong', 'trung h0.c ph0 th0ng', 'trunq h0c pk0 tk0nq', 
                        'trung học phổ thông', 'truong trung hoc pho thong', 'trường trung học phổ thông', 'tru0`ng trung h0.c ph0 th0ng',
                        'tru0nq trunq h0c pk0 tk0nq', 'truog trug hoc pho thog', 'trg thpt', 'trường thpt', 'truog trug hoc fo thog',
                        'cap 3', 'cấp 3', 'pho thong trung hoc', 'ph0 th0ng trung h0c', 'pk0 tk0nq trunq h0c',
                        'phổ thông trung học', 'fo thog trug hoc','truong pho thong trung hoc', 
                        'trường phổ thông trung học', 'truog fo thog trug hoc', 'tru0ng ph0 th0ng trung h0c', 'tru0nq pk0 tk0nq trunq h0c', 
                        'trường cấp 2-3', 'trường cap 3', 'trường cấp 3',
                        'trường cấp ba'],
                    'thpt': ['thpt', 'thptth', 'tkpt', 'tthpt','pt', 'ptth', 'pho thong', 'phổ thông', 'ph0 th0ng', 'pk0 tk0nq',
                        'trung hoc pho thong', 'trung h0.c ph0 th0ng', 'trunq h0c pk0 tk0nq', 
                        'trung học phổ thông', 'truong trung hoc pho thong', 'trường trung học phổ thông', 'tru0`ng trung h0.c ph0 th0ng',
                        'tru0nq trunq h0c pk0 tk0nq', 'truog trug hoc pho thog', 'trg thpt', 'trường thpt', 'truog trug hoc fo thog',
                        'cap 3', 'cấp 3', 'pho thong trung hoc', 'ph0 th0ng trung h0c', 'pk0 tk0nq trunq h0c',
                        'phổ thông trung học', 'fo thog trug hoc','truong pho thong trung hoc', 
                        'trường phổ thông trung học', 'truog fo thog trug hoc', 'tru0ng ph0 th0ng trung h0c', 'tru0nq pk0 tk0nq trunq h0c', 
                        'trường cấp 2-3', 'trường cap 3', 'trường cấp 3',
                        'trường cấp ba'],
                    'tp hcm': ['thành phố hcm', 'hồ chí minh', 'tp hcm', 'tp. hcm', 'hcm' 'tp.hcm', 'tp. ho chi minh', 'tp. hồ chí minh', 'tp. ho chj mjh', "tp. h0` chj' mjnh", 
                        'tphcm', 'tp ho chi minh', 'tp ho chj mjh', "tp h0` chj' mjnh", 'tp h0 ckj mjnk', 
                        'tp hồ chí minh', 'thành phố hồ chí minh', 'thah fo ho chj mjh', "tha`nh ph0' h0` chj' mjnh", 'tkank pk0 h0 ckj mjnk'],
                    'dh bach khoa ha noi': ['dai hoc bach khoa ha noi', 'đại học bách khoa hà nội', 'dhbkhn', 'dh bkhn', 
                        'daj hoc bax koa ha noj',
                        'daj h0c back kh0a ha n0j', 'dh bach khoa hn', 'dh bach khoa ha noi', 'dhbk ha noi', 'dhbk hn', 'bkhn',
                        'bk ha noi', 'đhbkhn', 'đh bkhn'],
                    'hv ngan hang': ['hoc vien ngan hang', 'học viện ngân hàng', 'hvnh', 'hoc vjen ngan hang', 'h0.c vjện ngân hàng', 'h0c vjen ngân hàng'],
                    'dh su pham': ['truong dai hoc su pham', 'trường đại học sư phạm', 'dhsp', 'dai hoc su pham',
                        'daj hoc su pham', 'truog daj hoc su fam', 'tru0nq daj h0c su pk4m',
                        'trg dhsp'],
                    'cd su pham': ['cao dang su pham', 'cdsp', 'cao dag su fam', 'ca0 danq su pkam', 'cao đẳng sư phạm',
                        'trường cao đẳng sư phạm', 'truong cao dang su pham', 'trg cdsp', 'truog cdsp', 'truog cao dag su fam', 
                        'tru0nq ca0 d4nq su pkam', 'cd sư phạm', 'cd sư phạm', 'cđ sư phạm'],
                    'giao thong van tai': ['giao thong van tai', 'giao thông vận tải', 'gtvt', 'jao thog van taj', 'gja0 th0ng van taj',
                        'qja0 tk0nq van taj', 'giao thong vt'],
                    'dh': ['dai hoc', 'đại học', 'daj hoc', 'dh', 'truong dai hoc', 'trường đại học', 'truong dh', 'truog daj hoc', 
                        'tru0nq daj h0c', 'daj h0c', 'trg dh', 'truog dh', 'dai hok', 'daj hok', 'đai hoc'],
                    'cd': ['cao dang', 'cao đẳng', 'cao dag', 'ca0 danq', 'cd', 'truong cao dang', 'trường cao đẳng', 'truog cao dag', 
                        'tru0nq ca0 danq', 'cđ'], 
                    'gdnn gdtx': ['giao duc nghe nghiep giao duc thuong xuyen', 'gdtx', 'ttgdtx', 'tt gdtx', 'ttgdnn gdtx',
                        'giao duc thuong xuyen', 'giáo dục thường xuyên', 'jao duc thuog xyen', 'gdnn gdtx', 'jao duc nge ngjep',
                        'trung tam gdtx', 'trung tâm gdtx', 'qja0 duc tku0nq xuyen', 'trug tam jao duc thuog xyen',
                        'trung tam giao duc thuong xuyen', 'trung tâm giáo dục thường xuyên', 'trunq tam qja0 duc tku0nq xuyen', 'trung tam gdnn gdtx',
                        'gdnn gdtx', 'trug tam gdtx', 'trunq tam gdtx', 'trunq tam gdnn gdtx',
                        'qja0 duc nqke nqkjep', 'jao duc nge ngjep jao duc thuog xyen',
                        'qja0 duc nqke nqkjep qja0 duc tku0nq xuyen'],
                    'gdtx': ['giao duc nghe nghiep giao duc thuong xuyen', 'gdtx', 'ttgdtx', 'tt gdtx', 'ttgdnn gdtx',
                        'giao duc thuong xuyen', 'giáo dục thường xuyên', 'jao duc thuog xyen', 'gdnn gdtx', 'jao duc nge ngjep',
                        'trung tam gdtx', 'trung tâm gdtx', 'qja0 duc tku0nq xuyen', 'trug tam jao duc thuog xyen',
                        'trung tam giao duc thuong xuyen', 'trung tâm giáo dục thường xuyên', 'trunq tam qja0 duc tku0nq xuyen', 'trung tam gdnn gdtx',
                        'gdnn gdtx', 'trug tam gdtx', 'trunq tam gdtx', 'trunq tam gdnn gdtx',
                        'qja0 duc nqke nqkjep', 'jao duc nge ngjep jao duc thuog xyen',
                        'qja0 duc nqke nqkjep qja0 duc tku0nq xuyen', 'trung tam giao duc nghe nghiep giao duc thuong xuyen', 
                        'tt giao duc nghe nghiep giao duc thuong xuyen'],
                    'hv': ['hoc vien', 'học viện', 'hoc vjen', 'h0.c vjện', 'h0c vjen', 'hv', 'HV'],
                    'dan lap': ['dan lap', 'dân lập' 'dl'],
                    'ptdt': ['ptdt nt', 'ptdtnt', 'pho thong dan toc noi tru', 'phổ thông dân tộc nội trú', 'ptdt noj tru' ,'fo thog dan toc noj tru',
                        'pk0 tk0nq dan t0c n0j tru'],
                    'bach khoa': ['bach khoa', 'bk', 'bách khoa', "bách's khoa's", 'back kh0a', 'back khoa'],
                    'quoc gia': ['quoc gia', 'qg', 'quoc gja', 'qgia', "quoc' gia", "quoc' gja", 'quốc gia', 'quốc gja'],
                    'nong nghiep': ['nong nghiep', 'nn', 'nogn nghiep', 'n0ng nghiep', 'nog nghjep', 'n0ng nghjep', 'nông nghiệp', 'nogn nghjep'],
                    
                    }

    for _ in range(num_iters):
        for k in augmentation.keys():
            if k in s_no_accents:
                index = s_no_accents.find(k)
                s_tmp = s_no_accents.replace(k, '', 1) # replace the first occurence
                n = random.randrange(len(augmentation[k]))
                insert_part = augmentation[k][n]
                one_case = s_tmp[:index] + insert_part + s_tmp[index:]
                if one_case not in results[s]:  
                    results[s].append(one_case)

def augmented_1(x):
    '''
        Used when using apply to a dataframe
        Input: 
            x: A string
        Output:
            result: A list of strings
    '''
    tmp1 = ['sinh viên ', 'sinh viên tại ', 'sjnh vjen ', 'sinh vjen ', 'sv ', 'sinh viên trường ', 'sv trường ', 'sv truog ', 'sinh vien truog ',
        'hội sinh viên ', 'hội sv ', 'hội sinh viên trường ', 'hoj sv ', 'hoi sjnh vien ', 'hoj sinh vjen ', 'hoi sjnh vjen ',
         'hội sv trường ', 'hội trai xinh gái đẹp ', 'hội trai xinh gái đẹp trường ', 'cộng đồng sinh viên trường ',
         'cộng đồng sinh viên ', 'hội cựu sinh viên trường ', 'hội cựu sinh viên ', 'confession trường ', 'confession ',
         'dự bị trường ', 'dự bị ', 'du bj ', 'sinh vjen ', 'hoj cuu sv ', 'hội cựu sv ', 'hsv ', 'hoc tai ', 'học tại ',
           'hoc taj ', 'hok tai ', 'hok taj '] # Dai hoc, cao dang, hoc vien

    tmp2 = ['du lịch ', 'du lich ', 'quốc tế ', 'quoc te ', 'marketing ', 'truyền hình ', 'truyen hinh ', 'truyen hjnh ',
            'y ', 'luật ', 'luat ', 'lý ', 'ly ', 'ngoại ngữ ', 'ngoai ngu ', 'ngoaj ngu ', 'điều dưỡng ', 'dieu duong ',
            'djeu duong ', 'kinh tế và kinh doanh quốc tế ', 'luật kinh tế ', 'luat kinh te ', 'luat kjnh te ',
            'tài chính ngân hàng ', 'taj chinh ngan hang ', 'tai chinh nh ', 'tcnh ', 'dược ', 'duoc ', 
            'điện tử thông tin ', 'djen tu thong tin ', 'dien tu thong tin ', 'dien tu thong tjn ',
        'giáo dục tiểu học ', 'giao duc tieu hoc ', 'gjao duc tjeu hoc ', 'giáo dục thể chất ', 'giao duc the chat ',
            'quản trị văn phòng ', 'quan tri van phong ', 'hàn ', 'han ',  'công nghệ thông tin ', 'cntt ', 'cong nghe thong tin ',
            'cong nghe thong tjn ', 'tiếng trung ', 'tieng trung ', 'tjeng trung ', 'y dược ', 'y duoc ', 'nông học ', 'nong hoc ',
            'khoa học xã hội và nhân văn ', 'khoa hoc xa hoi va nhan van ', 'khxh & nv ', 'thống kê tin học ', 'thong ke tin hoc ', 
        'tổ chức và quản lý nhân lực ', 'to chuc va quan ly nhan luc ', 'thủy sản ', 'thuy san ', 'tài chính ', 'tai chinh ',
            'cơ khí chế tạo máy ', 'co khi che tao may ', 'lâm nghiệp ', 'lam nghiep ', 'kinh tế ', 'kinh te ', 'kjnh te ',
            'xây dựng ', 'xay dung ', 'công nghệ ', 'cong nghe ', 'sư phạm kỹ thuật ', 'spkt ', 'su pham ky thuat ',
            'vật lý ', 'vat ly ', 'toán ', 'toan ', 'hóa ', 'hoa ', 'kế toán ứng dụng ', 'ke toan ung dung ',
        'kế toán ', 'ke toan ', 'quản trị kinh doanh ', 'quan tri kinh doanh ', 'việt nam học ', 'viet nam hoc ', 'vjet nam hoc ',
            'văn học ', 'van hoc ', 'ngôn ngữ anh ', 'ngon ngu anh ', 'nuôi trồng thủy sản ', 'nuoi trong thuy san ',
        'quản lý tài nguyên và môi trường ', 'quan ly tai nguyen va moi truong ', 'quản lý đất đai ', 'quan ly dat dai ', 
            'công nghệ thực phẩm ', 'cong nghe thuc pham ', 'thú y ', 'thu y ', 'chăn nuôi ', 'chan nuoi ', 'chan nuoj ',
        'quản trị khách sạn ', 'quan tri khach san ', 'quan trj khach san '] # Khoa, Vien

    tmp3 = ['điện ', 'dien ', 'djen ', 'toán tin ', 'toan tin ', 'toan tjn ', 'điện tử viễn thông ', 'dtvt ', 'đtvt ', 'djen tu vien thong ',
            'dien tu vien thong ', 'công nghệ thông tin ', 'cong nghe thong tin ', 'cntt ', 'cong nghe thong tjn ', 'đào tạo quốc tế ', 
            'dao tao quoc te ', 'cơ khí ', 'co khi ', 'ngoại ngữ ', 'ngoai ngu ', 'ngoaj ngu ', 'kinh tế ', 'kinh te ', 'kjnh te ']
    tmp4 = ['gia sử ', 'gia sư ', 'gia su ', 'gja su ', 'tiếng trung ', 'tieng trung ', 'tiếng anh ', 'tieng anh ', 'tiếng nhật ', 'tieng nhat ', 'tiếng hàn ',
            'tieng han ', 'tiếng đức ', 'tieng duc '] # Cau lac bo

    tmp5 = ['chuyên toán ', 'chuyen toan ', 'chuyên lý ', 'chuyen ly ', 'chuyên hóa ', 'chuyen hoa ', 'chuyên tin ', 'chuyen tin ', 'chuyên anh ',
        'chuyen anh ', 'chuyên văn ', 'chuyen van ', 'chuyên sử ', 'chuyen su ', 'chuyên địa ', 'chuyen dia ', 'chuyên sinh ', 'chuyen sinh ',
        'lớp 12 chuyên toán ', 'lớp 11 chuyên toán ', 'lớp 10 chuyên toán ', 'lớp 12 chuyên lý ', 'lớp 11 chuyên lý ', 'lớp 10 chuyên lý ',
        'lớp 12 chuyên hóa ', 'lớp 11 chuyên hóa ', 'lớp 10 chuyên hóa ', 'lớp 12 chuyên tin ', 'lớp 11 chuyên tin ', 'lớp 10 chuyên tin ',
        'lớp 12 chuyên anh ', 'lớp 11 chuyên anh ', 'lớp 10 chuyên anh ', 'lớp 12 chuyên văn ', 'lớp 11 chuyên văn ', 'lớp 10 chuyên văn ',
        'lớp 12 chuyên sử ', 'lớp 11 chuyên sử ', 'lớp 10 chuyên sử ', 'lớp 12 chuyên địa ', 'lớp 11 chuyên địa ', 'lớp 10 chuyên địa ',
        'lớp 12 chuyên sinh ', 'lớp 11 chuyên sinh ', 'lớp 10 chuyên sinh ', 'lớp 12 khối chuyên ', 'lớp 11 khối chuyên ', 'lớp 10 khối chuyên '] # Lop, THPT chuyen

    tmp6 = ['lớp 12a1 ', 'lop 12a1 ', 'lớp 12 ', 'lop 12 ', 'học sinh tại ', 'hoc sinh tai ', 'confession ', 'hok sinh ', 'hoc sink ',
        'hok sink ', 'hoc tai ', 'học tại ', 'hoc taj ', 'hok tai ', 'hok taj ', 'lop 12a2 ', 'lớp 12a2 ', 'lớp 12a3 ', 'lop 12a3 ',
        'lớp 12a4 ', 'lop 12a4 ', 'lớp 12a5 ', 'lop 12a5 ', 'lớp 12a6 ', 'lop 12a6 ', 'lớp 12a7 ', 'lop 12a7 ', 'lớp 11 ', 'lop 11 ', 
        'lớp 11a1 ', 'lớp 11a2 ', 'lớp 11a3 ', 'lớp 11a4 ', 'lớp 11a5 ', 'lớp 11a6 ', 'lớp 11a7 ', 'lop 11a1 ', 'lop 11a2 ',
        'lop 11a3 ', 'lop 11a4 ', 'lop 11a5 ', 'lop 11a6 ', 'lop 11a7 ', 'lớp 10 ', 'lop 10 ', 
        'lớp 10a1 ', 'lớp 10a2 ', 'lớp 10a3 ', 'lớp 10a4 ', 'lớp 10a5 ', 'lớp 10a6 ', 'lớp 10a7 ', 'lop 10a1 ', 'lop 10a2 ',
        'lop 10a3 ', 'lop 10a4 ', 'lop 10a5 ', 'lop 10a6 ', 'lop 10a7 '] # Lop, THPT thuong

    result = list()
    x = x.lower()
    if re.match('^đh', x) or re.match('^cđ', x) or re.match('^hv', x):
        for i in random.choices(tmp1, k=round(0.7*len(tmp1))):
            if i == 'confession ' or i == 'confession trường ':
                result.append(remove_accent_randomly(text=x + ' ' + 'confession'))
            result.append(remove_accent_randomly(text=i + x))
        for i in random.choices(tmp2, k=round(0.7*len(tmp2))):
            result.append(remove_accent_randomly(text='khoa ' + i + x))
        for i in random.choices(tmp3, k=round(0.7*len(tmp3))):
            result.append(remove_accent_randomly(text='viện ' + i + x))
            result.append(remove_accent_randomly(text='vjen ' + i + x))
        
        for i in random.choices(tmp4, k=round(0.7*len(tmp4))):
            result.append(remove_accent_randomly(text='câu lạc bộ ' + i + x))
            result.append(remove_accent_randomly(text='clb ' + i + x))
    elif ('thpt' in x) and (('chuyên' in x) or ('năng khiếu' in x)) :
        for i in random.choices(tmp5, k=round(0.7*len(tmp5))):
            result.append(remove_accent_randomly(text=i + x))
    elif re.match('^thpt', x) or re.match('^thcs-thpt', x):
        for i in random.choices(tmp6, k=round(0.7*len(tmp6))):
            if i == 'confession ':
                result.append(remove_accent_randomly(text=x + ' ' + 'confession'))
            remove_accent_randomly(text=' '.join(x.split()[1:]) + ' high school')
            result.append(remove_accent_randomly(text=' '.join(x.split()[1:]) + ' highschool'))
            result.append(remove_accent_randomly(text=' '.join(x.split()[1:]) + ' high school'))
            result.append(remove_accent_randomly(text=i + x))
    
    return result

def randomly_select(x, m, n):
    '''
        Used when using apply to a dataframe
        Input: 
            x: A row of the dataframe
            m: The number of randomly chosen elements if the string contains `gdtx`
            n: The number of randomly chosen elements if the string doesn't contain 'gdtx' (means Dai hoc, cao dang, hoc vien,...)
        Output:
            A Series representing a row of the dataframe
    '''
    random.shuffle(x['augmented'])
    if 'gdtx' in x['unique_name']:
        if len(x['augmented']) >= m:
            x['augmented'] = random.choices(x['augmented'], k=m) # select 6 random elements if the length of the list is greater than or equal 6
            return x
        else:
            return x
    else:
        if len(x['augmented']) >= n:
            x['augmented'] = random.choices(x['augmented'], k=n) # select 9 random elements if the length of the list is greater than or equal 9
            return x
        else:
            return x


if __name__ == '__main__':
    train_path = '/home/sonnh/son/VND/augmenting-education/data_for_augmentation' # path to training data, contains `text`, `normed_text` columns
    df = pd.read_csv(os.path.join(train_path, 'final_train_old.csv'))
    
    df['lower_text'] = df['text'].apply(lambda x: x.lower())

    # Add lower_text to tmp_dict
    tmp_dict = dict()
    for x in tqdm(df['normed_text']):
        tmp_dict[x] = df.loc[df['normed_text']==x , 'lower_text'].to_list()

    df2 = pd.DataFrame(df['normed_text'].unique(), columns=['unique_name']) # contains unique names of column `normed_text` in df
    
    df2['augmented_1'] = df2['unique_name'].progress_apply(lambda x: augmented_1(x)) # Them khoa, vien, lop, ... trong cot `augmented_1`

    for i in tqdm(range(len(df2)), desc="Augmentation"):
        text = df2.loc[i, 'unique_name']
        generate_augmentation(text, num_iters=1000) # 'results' is changed
    
    # Add tmp_dict to results
    for k in results.keys():
        if k in tmp_dict.keys():
            for i in tmp_dict[k]:
                if i in results[k]:
                    results[k].append(i)

    df2['augmented_2'] = results.values() # create column `augmented_2` in df2
    df2['augmented'] = df2['augmented_1'] + df2['augmented_2'] # create column `augmented` in df2
    
    df2 = df2.apply(lambda x: randomly_select(x, m=6, n=9), axis=1) # changes column `augmented`
    df2['count'] = df2['augmented'].apply(lambda x: len(x)) # create column `count` in df2

    df2['tmp'] = df2.apply(lambda x: [x['unique_name']]*x['count'], axis=1) # create column `tmp`. It's the label data and used to flatten a list of lists later

    # Flattening
    flat_list1 = list(itertools.chain(*df2['augmented'].values))
    flat_list2 = list(itertools.chain(*df2['tmp'].values))

    df3 = pd.DataFrame({'normed_text': flat_list2, 'augmented': flat_list1})
    df3 = df3.drop_duplicates(keep='first') # drop duplicate rows in df3, keep the first occurence

    # Saving augmented results
    augmented_path = './data_augmented'
    df3.to_csv(os.path.join(augmented_path, 'final_train_old_augmented.csv'), index=False)


    

    
    
            


