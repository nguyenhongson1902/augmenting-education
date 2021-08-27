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
    
    augmentation = {'thpt': ['thpt', 'tthpt','pt', 'ptth', 'pho thong', 'trung hoc pho thong', 'truong trung hoc pho thong', 'cap 3', 'cap 3', 'pho thong trung hoc', 'truong pho thong trung hoc'],
                    'high school': ['high school', 'highschool'],
                    'truong trung hoc pho thong': ['truong trung hoc pho thong', 'thpt', 'pt', 'ptth', 'pho thong', 'trung hoc pho thong', 'truong thpt', 'truong cap 3', 'cap 3', 'pho thong trung hoc'],
                    'truong thpt': ['truong thpt', 'thpt', 'pt', 'ptth', 'pho thong', 'trung hoc pho thong', 'truong trung hoc pho thong', 'truong cap 3', 'cap 3', 'pho thong trung hoc'],
                    'trung hoc pho thong': ['trung hoc pho thong', 'thpt', 'pt', 'ptth', 'pho thong', 'truong thpt', 'truong trung hoc pho thong', 'truong cap 3', 'cap 3', 'pho thong trung hoc'],
                    'tp hcm': ['tp hcm', 'tp.hcm', 'tp. ho chi minh', 'tphcm', 'tp ho chi minh'],
                    'tp ho chi minh': ['tp ho chi minh', 'tp hcm', 'tp.hcm', 'tp. ho chi minh', 'tphcm'],
                    'tp. ho chi minh': ['tp. ho chi minh', 'tp hcm', 'tp.hcm', 'tp ho chi minh', 'tphcm'],
                    'dai hoc': ['dai hoc', 'daj hoc', 'dh', 'truong dai hoc', 'truong dh'],
                    'truong dai hoc': ['truong dai hoc', 'dai hoc', 'daj hoc', 'dh', 'truong dh'],
                    'cao dang': ['cao dang', 'cd', 'truong cao dang'],
                    'truong cao dang': ['truong cao dang', 'cd', 'cao dang'],
                    'cao dang su pham': ['cao dang su pham', 'cdsp'],
                    'truong cao dang su pham': ['truong cao dang su pham', 'cao dang su pham', 'cdsp'],
                    'dan lap': ['dan lap', 'dl'],
                    'ptdt nt': ['ptdt nt', 'ptdtnt', 'pho thong dan toc noi tru'],
                    'ptdt bt': ['ptdt bt', 'ptdtbt', 'pho thong dan toc ban tru'],
                    'gdtx': ['gdtx', 'ttgdtx', 'giao duc thuong xuyen', 'trung tam gdtx', 'trung tam giao duc thuong xuyen'],
                    'trung tam gdtx': ['trung tam gdtx', 'gdtx', 'ttgdtx', 'giao duc thuong xuyen', 'trung tam giao duc thuong xuyen'],
                    'trung tam giao duc thuong xuyen': ['trung tam giao duc thuong xuyen', 'trung tam gdtx', 'gdtx', 'ttgdtx', 'giao duc thuong xuyen'],
                    'giao duc thuong xuyen': ['giao duc thuong xuyen', 'gdtx', 'ttgdtx', 'trung tam gdtx', 'trung tam giao duc thuong xuyen'],
                    'bach khoa': ['bach khoa', 'bk'],
                    'dai hoc bach khoa ha noi': ['dai hoc bach khoa ha noi', 'dhbkhn'],
                    'hoc vien ngan hang': ['hoc vien ngan hang', 'hvnh'],
                    'giao thong van tai': ['giao thong van tai', 'gtvt'],
                    'truong dai hoc su pham': ['truong dai hoc su pham', 'dhsp', 'dai hoc su pham'],
                    'dai hoc su pham': ['dai hoc su pham', 'truong dai hoc su pham', 'dhsp']}

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
    df = df.dropna()
    df = df.reset_index(drop=True)

    df2 = pd.DataFrame(df['normed_text'].unique(), columns=['normed_text_unique'])


    for i in tqdm(range(len(df2)), desc="Augmentation"):
        text = df2['normed_text_unique'][i]
        generate_augmentation(text, num_iters=1000) # 'results' is changed

    df2['augmented_name'] = df2['normed_text_unique'].apply(lambda x: results[x])
    df2['count'] = df2['normed_text_unique'].apply(lambda x: len(results[x]))

    df2.to_csv('D:\Projects\VND_work\classification_fb_education\data_augmented\education_train_count.csv', index=False)


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
            


