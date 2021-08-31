import unidecode 
import random
import re
import pandas as pd
from tqdm import tqdm


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

def generate_augmentation(s, num_iters=1000):
    '''
        Create fake data for education. Every iteration, we pick a random case in the dictionary `augmentation`.
        This function will change the dictionary `results`
        Input:
            s: str
            num_iters: The number of iterations (default=1000)
    '''
    global results # storage
    s_no_accents = remove_accent(s).lower()
    results[s] = [s, s_no_accents]
    
    augmentation = {'truong trung hoc pho thong': ['thpt', 'thptth', 'tkpt', 'tthpt','pt', 'ptth', 'pho thong', 'phổ thông', 'ph0 th0ng', 'pk0 tk0nq',
                        'trung hoc pho thong', 'trung h0.c ph0 th0ng', 'trunq h0c pk0 tk0nq', 
                        'trung học phổ thông' 'truong trung hoc pho thong', 'trường trung học phổ thông', 'tru0`ng trung h0.c ph0 th0ng',
                        'tru0nq trunq h0c pk0 tk0nq', 'truog trug hoc pho thog', 'trg thpt', 'trường thpt', 'truog trug hoc fo thog',
                        'cap 3', 'cấp 3', 'c4p 3', 'pho thong trung hoc', 'ph0 th0ng trung h0c', 'pk0 tk0nq trunq h0c',
                        'phổ thông trung học', 'fo thog trug hoc','truong pho thong trung hoc', 
                        'trường phổ thông trung học', 'truog fo thog trug hoc', 'tru0ng ph0 th0ng trung h0c', 'tru0nq pk0 tk0nq trunq h0c'],
                    'trung hoc pho thong': ['thpt', 'thptth', 'tkpt', 'tthpt','pt', 'ptth', 'pho thong', 'phổ thông', 'ph0 th0ng', 'pk0 tk0nq',
                        'trung hoc pho thong', 'trung h0.c ph0 th0ng', 'trunq h0c pk0 tk0nq', 
                        'trung học phổ thông' 'truong trung hoc pho thong', 'trường trung học phổ thông', 'tru0`ng trung h0.c ph0 th0ng',
                        'tru0nq trunq h0c pk0 tk0nq', 'truog trug hoc pho thog', 'trg thpt', 'trường thpt', 'truog trug hoc fo thog',
                        'cap 3', 'cấp 3', 'c4p 3', 'pho thong trung hoc', 'ph0 th0ng trung h0c', 'pk0 tk0nq trunq h0c',
                        'phổ thông trung học', 'fo thog trug hoc','truong pho thong trung hoc', 
                        'trường phổ thông trung học', 'truog fo thog trug hoc', 'tru0ng ph0 th0ng trung h0c', 'tru0nq pk0 tk0nq trunq h0c'],
                    'truong thpt': ['thpt', 'thptth', 'tkpt', 'tthpt','pt', 'ptth', 'pho thong', 'phổ thông', 'ph0 th0ng', 'pk0 tk0nq',
                        'trung hoc pho thong', 'trung h0.c ph0 th0ng', 'trunq h0c pk0 tk0nq', 
                        'trung học phổ thông' 'truong trung hoc pho thong', 'trường trung học phổ thông', 'tru0`ng trung h0.c ph0 th0ng',
                        'tru0nq trunq h0c pk0 tk0nq', 'truog trug hoc pho thog', 'trg thpt', 'trường thpt', 'truog trug hoc fo thog',
                        'cap 3', 'cấp 3', 'c4p 3', 'pho thong trung hoc', 'ph0 th0ng trung h0c', 'pk0 tk0nq trunq h0c',
                        'phổ thông trung học', 'fo thog trug hoc','truong pho thong trung hoc', 
                        'trường phổ thông trung học', 'truog fo thog trug hoc', 'tru0ng ph0 th0ng trung h0c', 'tru0nq pk0 tk0nq trunq h0c'],
                    'thpt': ['thpt', 'thptth', 'tkpt', 'tthpt','pt', 'ptth', 'pho thong', 'phổ thông', 'ph0 th0ng', 'pk0 tk0nq',
                        'trung hoc pho thong', 'trung h0.c ph0 th0ng', 'trunq h0c pk0 tk0nq', 
                        'trung học phổ thông' 'truong trung hoc pho thong', 'trường trung học phổ thông', 'tru0`ng trung h0.c ph0 th0ng',
                        'tru0nq trunq h0c pk0 tk0nq', 'truog trug hoc pho thog', 'trg thpt', 'trường thpt', 'truog trug hoc fo thog',
                        'cap 3', 'cấp 3', 'c4p 3', 'pho thong trung hoc', 'ph0 th0ng trung h0c', 'pk0 tk0nq trunq h0c',
                        'phổ thông trung học', 'fo thog trug hoc','truong pho thong trung hoc', 
                        'trường phổ thông trung học', 'truog fo thog trug hoc', 'tru0ng ph0 th0ng trung h0c', 'tru0nq pk0 tk0nq trunq h0c'],
                    'high school': ['high school', 'highschool'],
                    'tp. ho chi minh': ['thành phố hcm', 'hồ chí minh', 'tp hcm', 'tp. hcm', 'hcm' 'tp.hcm', 'tp. ho chi minh', 'tp. hồ chí minh', 'tp. ho chj mjh', "tp. h0` chj' mjnh", 
                        'tphcm', 'tp ho chi minh', 'tp ho chj mjh', "tp h0` chj' mjnh", 'tp h0 ckj mjnk', 
                        'tp hồ chí minh', 'thành phố hồ chí minh', 'thah fo ho chj mjh', "th4`nh ph0' h0` chj' mjnh", 'tk4nk pk0 h0 ckj mjnk'],
                    'tp ho chi minh': ['thành phố hcm', 'hồ chí minh', 'tp hcm', 'tp. hcm', 'hcm' 'tp.hcm', 'tp. ho chi minh', 'tp. hồ chí minh', 'tp. ho chj mjh', "tp. h0` chj' mjnh", 
                        'tphcm', 'tp ho chi minh', 'tp ho chj mjh', "tp h0` chj' mjnh", 'tp h0 ckj mjnk', 
                        'tp hồ chí minh', 'thành phố hồ chí minh', 'thah fo ho chj mjh', "th4`nh ph0' h0` chj' mjnh", 'tk4nk pk0 h0 ckj mjnk'],
                    'tp hcm': ['thành phố hcm', 'hồ chí minh', 'tp hcm', 'tp. hcm', 'hcm' 'tp.hcm', 'tp. ho chi minh', 'tp. hồ chí minh', 'tp. ho chj mjh', "tp. h0` chj' mjnh", 
                        'tphcm', 'tp ho chi minh', 'tp ho chj mjh', "tp h0` chj' mjnh", 'tp h0 ckj mjnk', 
                        'tp hồ chí minh', 'thành phố hồ chí minh', 'thah fo ho chj mjh', "th4`nh ph0' h0` chj' mjnh", 'tk4nk pk0 h0 ckj mjnk'],
                    'hcm': ['thành phố hcm', 'hồ chí minh', 'tp hcm', 'tp. hcm', 'hcm' 'tp.hcm', 'tp. ho chi minh', 'tp. hồ chí minh', 'tp. ho chj mjh', "tp. h0` chj' mjnh", 
                        'tphcm', 'tp ho chi minh', 'tp ho chj mjh', "tp h0` chj' mjnh", 'tp h0 ckj mjnk', 
                        'tp hồ chí minh', 'thành phố hồ chí minh', 'thah fo ho chj mjh', "th4`nh ph0' h0` chj' mjnh", 'tk4nk pk0 h0 ckj mjnk'],
                    'truong dai hoc': ['dai hoc', 'đại học', 'daj hoc', 'dh', 'truong dai hoc', 'trường đại học', 'truong dh', 'truog daj hoc', 
                        'tru0`ng +)4j h0.c', 'tru0nq d4j h0c', '+)4j h0.c', "đại's học's", 'd4j h0c', 'trg dh', 'truog dh', 'dai hok', 'daj hok'],
                    'dai hoc': ['dai hoc', 'đại học', 'daj hoc', 'dh', 'truong dai hoc', 'trường đại học', 'truong dh', 'truog daj hoc', 
                        'tru0`ng +)4j h0.c', 'tru0nq d4j h0c', '+)4j h0.c', "đại's học's", 'd4j h0c', 'trg dh', 'truog dh', 'dai hok', 'daj hok'],
                    'truong cao dang su pham': ['cao dang su pham', 'cdsp', 'cao dag su fam', 'c40 ])4ng su ph4m', 'c40 d4nq su pk4m', 'cao đẳng sư phạm',
                        'trường cao đẳng sư phạm', 'truong cao dang su pham', 'trg cdsp', 'truog cdsp', 'truog cao dag su fam', 
                        'tru0`ng c40 +)4ng su ph4m', 'tru0nq c40 d4nq su pk4m'],
                    'cao dang su pham': ['cao dang su pham', 'cdsp', 'cao dag su fam', 'c40 ])4ng su ph4m', 'c40 d4nq su pk4m', 'cao đẳng sư phạm',
                        'trường cao đẳng sư phạm', 'truong cao dang su pham', 'trg cdsp', 'truog cdsp', 'truog cao dag su fam', 
                        'tru0`ng c40 +)4ng su ph4m', 'tru0nq c40 d4nq su pk4m'],
                    'truong cao dang': ['cao dang', 'cao đẳng', 'cao dag', 'c40 +)4ng', 'c40 d4nq', 'cd', 'truong cao dang', 'trường cao đẳng', 'truog cao dag', 
                        'tru0`ng c40 +)4ng', 'tru0nq c40 d4nq'],
                    'cao dang': ['cao dang', 'cao đẳng', 'cao dag', 'c40 +)4ng', 'c40 d4nq', 'cd', 'truong cao dang', 'trường cao đẳng', 'truog cao dag', 
                        'tru0`ng c40 +)4ng', 'tru0nq c40 d4nq'],
                    'hoc vien': ['hoc vien', 'học viện', 'hoc vjen', 'h0.c vjện', 'h0c vj3n'],
                    'dan lap': ['dan lap', 'dân lập' 'dl', 'd4n l4p'],
                    'ptdt': ['ptdt nt', 'ptdtnt', 'pho thong dan toc noi tru', 'phổ thông dân tộc nội trú', 'ptdt noj tru' ,'fo thog dan toc noj tru',
                        'pk0 tk0nq d4n t0c n0j tru'],
                    'trung tam giao duc thuong xuyen': ['gdtx', 'ttgdtx', 'tt gdtx', 'ttgdnn gdtx' 'giao duc thuong xuyen', 'giáo dục thường xuyên', 'jao duc thuog xyen',
                        'trung tam gdtx', 'trung tâm gdtx', 'qj40 duc tku0nq xuy3n', 'trug tam jao duc thuog xyen', "trung t4m gj4'0 ])u.c thu0`ng xuij3n",
                        'trung tam giao duc thuong xuyen', 'trung tâm giáo dục thường xuyên', 'trunq t4m qj40 duc tku0nq xuy3n', 'trung tam gdnn gdtx',
                        'gdnn gdtx', 'trug tam gdtx', 'trunq t4m gdtx', 'trunq t4m gdnn gdtx', 'trunq t4m gdnn gdtx'],
                    'giao duc thuong xuyen': ['gdtx', 'ttgdtx', 'tt gdtx', 'ttgdnn gdtx' 'giao duc thuong xuyen', 'giáo dục thường xuyên', 'jao duc thuog xyen',
                        'trung tam gdtx', 'trung tâm gdtx', 'qj40 duc tku0nq xuy3n', 'trug tam jao duc thuog xyen', "trung t4m gj4'0 ])u.c thu0`ng xuij3n",
                        'trung tam giao duc thuong xuyen', 'trung tâm giáo dục thường xuyên', 'trunq t4m qj40 duc tku0nq xuy3n', 'trung tam gdnn gdtx',
                        'gdnn gdtx', 'trug tam gdtx', 'trunq t4m gdtx', 'trunq t4m gdnn gdtx', 'trunq t4m gdnn gdtx'],
                    'trung tam gdtx': ['gdtx', 'ttgdtx', 'tt gdtx', 'ttgdnn gdtx' 'giao duc thuong xuyen', 'giáo dục thường xuyên', 'jao duc thuog xyen',
                        'trung tam gdtx', 'trung tâm gdtx', 'qj40 duc tku0nq xuy3n', 'trug tam jao duc thuog xyen', "trung t4m gj4'0 ])u.c thu0`ng xuij3n",
                        'trung tam giao duc thuong xuyen', 'trung tâm giáo dục thường xuyên', 'trunq t4m qj40 duc tku0nq xuy3n', 'trung tam gdnn gdtx',
                        'gdnn gdtx', 'trug tam gdtx', 'trunq t4m gdtx', 'trunq t4m gdnn gdtx', 'trunq t4m gdnn gdtx'],
                    'gdtx': ['gdtx', 'ttgdtx', 'tt gdtx', 'ttgdnn gdtx' 'giao duc thuong xuyen', 'giáo dục thường xuyên', 'jao duc thuog xyen',
                        'trung tam gdtx', 'trung tâm gdtx', 'qj40 duc tku0nq xuy3n', 'trug tam jao duc thuog xyen', "trung t4m gj4'0 ])u.c thu0`ng xuij3n",
                        'trung tam giao duc thuong xuyen', 'trung tâm giáo dục thường xuyên', 'trunq t4m qj40 duc tku0nq xuy3n', 'trung tam gdnn gdtx',
                        'gdnn gdtx', 'trug tam gdtx', 'trunq t4m gdtx', 'trunq t4m gdnn gdtx', 'trunq t4m gdnn gdtx'],
                    'dai hoc bach khoa ha noi': ['dai hoc bach khoa ha noi', 'đại học bách khoa hà nội', 'dhbkhn', 'dh bkhn', 
                        'daj hoc bax koa ha noj', "+)4j h0.c |34'ch kh04 h4` n0.j", "đại's học's bách's khoa's hà's nội's",
                        'd4j h0c b4ck kh04 h4 n0j', 'dh bach khoa hn', 'dh bach khoa ha noi', 'dhbk ha noi', 'dhbk hn', 'bkhn',
                        'bk ha noi'],
                    'dh bach khoa ha noi': ['dai hoc bach khoa ha noi', 'đại học bách khoa hà nội', 'dhbkhn', 'dh bkhn', 
                        'daj hoc bax koa ha noj', "+)4j h0.c |34'ch kh04 h4` n0.j", "đại's học's bách's khoa's hà's nội's",
                        'd4j h0c b4ck kh04 h4 n0j', 'dh bach khoa hn', 'dh bach khoa ha noi', 'dhbk ha noi', 'dhbk hn', 'bkhn',
                        'bk ha noi'],
                    'bach khoa': ['bach khoa', 'bk', 'bách khoa', "bách's khoa's", 'b4ck kh04', 'back khoa'],
                    'hoc vien ngan hang': ['hoc vien ngan hang', 'học viện ngân hàng', 'hvnh'],
                    'giao thong van tai': ['giao thong van tai', 'giao thông vận tải', 'gtvt', 'jao thog van taj', 'gj40 th0ng v4n t4j',
                        'qj40 tk0nq v4n t4j', 'giao thong vt'],
                    'truong dai hoc su pham': ['truong dai hoc su pham', 'trường đại học sư phạm', 'dhsp', 'dai hoc su pham',
                        'daj hoc su pham', 'truog daj hoc su fam', 'tru0`ng +)4j h0.c su ph4m', 'tru0nq d4j h0c su pk4m',
                        'trg dhsp'],
                    'dai hoc su pham': ['truong dai hoc su pham', 'trường đại học sư phạm', 'dhsp', 'dai hoc su pham',
                        'daj hoc su pham', 'truog daj hoc su fam', 'tru0`ng +)4j h0.c su ph4m', 'tru0nq d4j h0c su pk4m',
                        'trg dhsp']}

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
                

if __name__ == '__main__':
    path = 'D:\Projects\VND_work\classification_fb_education\data_for_augmentation\education_train.csv'

    df = pd.read_csv(path)
    df['lower_text'] = df['text'].apply(lambda x: x.lower())
    df['lower_normed_text'] = df['normed_text'].apply(lambda x: x.lower())
    
    df = df.dropna()
    df = df.reset_index(drop=True)

    df2 = pd.DataFrame(df['lower_text'].unique(), columns=['lower_text_unique'])


    for i in tqdm(range(len(df2)), desc="Augmentation"):
        text = df2['lower_text_unique'][i]
        generate_augmentation(text, num_iters=1000) # 'results' is changed

    df2['augmented_name'] = df2['lower_text_unique'].apply(lambda x: results[x])
    df2['count'] = df2['lower_text_unique'].apply(lambda x: len(results[x]))

    # Adding original data to fake data
    for i in tqdm(range(len(df2)), desc="Adding Original to Fake"):
        content = df2.loc[i, 'lower_text_unique']
        original_list = df.loc[df['lower_text']==content, :]['lower_text'].to_list()
        df2.loc[i, 'augmented_name'].extend(original_list)
        df2.loc[i, 'count'] += len(original_list)
    
    # Save df2
    df2.to_csv('D:\Projects\VND_work\classification_fb_education\data_augmented\education_train_count.csv', index=False)

    # Flattening
    n_rows = df2['count'].sum()
    n_columns = 2
    df3 = pd.DataFrame(index=range(n_rows), columns=range(n_columns))
    i = 0

    for k, v in tqdm(results.items(), desc='Flattening'):
        for element in v:
            df3[0][i] = k
            df3[1][i] = element
            i += 1
    df3.columns = ['normed_text', 'original_and_fake']

    file_path = 'D:\Projects\VND_work\classification_fb_education\data_augmented\education_train_augmented.csv'
    df3.to_csv(file_path, index=False)
            


