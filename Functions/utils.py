# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 18:43:11 2023

@author: pathouli
"""

def clean_text(str_in):
    import re
    tmp = re.sub("[^A-Za-z']+", " ",str_in).lower().strip().replace("  ", " ")
    return tmp

def file_clean(path_in):
    f = open(path_in, encoding="UTF-8")
    tmp = f.read()
    f.close()
    tmp = clean_text(tmp)
    return tmp

def read_files(path_in):
    import os
    import pandas as pd
    file_list = pd.DataFrame()
    for root, dirs, files in os.walk(path_in, topdown=False):
        for name in files:
            try:
                t_path = root + "/" + name
                file_p = file_clean(t_path)
                t_p = root.split("/")[-1:][0]
                if len(file_p) > 0:
                    file_list = file_list.append(
                        {"body": file_p, "label": t_p
                         }, ignore_index=True)
            except:
                print (t_path)
                pass
    return file_list

def wrd_dictionary(df_in, col_name_in):
    import collections
    my_dictionaty_t = dict()
    for topic_t in df_in.label.unique():
        tmp = df_in[df_in.label == topic_t]
        tmp = tmp[col_name_in].str.cat(sep=" ")
        wrd_freq = collections.Counter(tmp.split())
        my_dictionaty_t[topic_t] = wrd_freq
    return my_dictionaty_t

def rem_sw(var_in):
    from nltk.corpus import stopwords
    sw = stopwords.words("english")
    tmp = var_in.split()
    # tmp_ar = list()
    # for word_t in tmp:
    #     if word_t not in sw:
    #         tmp_ar.append(word_t)
    tmp_ar = [word_t for word_t in tmp if word_t not in sw]
    tmp_o = ' '.join(tmp_ar)
    return tmp_o

def write_pickle(obj_in, path_in, name_in):
    import pickle
    pickle.dump(obj_in, open(path_in + name_in + ".pk", 'wb'))
    
def read_pickle(path_in, name_in):
    import pickle
    the_data_t = pickle.load(open(path_in + name_in + ".pk", 'rb'))
    return the_data_t

def wrd_cnt(txt_in):
    tmp = len(set(txt_in.split()))
    return tmp

def stem_fun(txt_in):
    from nltk.stem import PorterStemmer
    stem_tmp = PorterStemmer()
    tmp = [stem_tmp.stem(word) for word in txt_in.split()]
    tmp = ' '.join(tmp)
    # tmp = list()
    # for word in txt_in.split():
    #     tmp.append(stem_tmp.stem(word))
    return tmp

def vec_fun(df_in, m_in, n_in, name_in, out_p_in):
    #turn into a function called vec_fun and give user ability to set arbitrary ngrams
    #and an arbitrary name for the saved pk object
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    import pandas as pd
    if name_in == "vec":
        xform = CountVectorizer(ngram_range=(m_in, n_in))
    else:
        xform = TfidfVectorizer(ngram_range=(m_in, n_in))
    xform_data = pd.DataFrame(xform.fit_transform(df_in).toarray()) #memory hog
    xform_data.columns = xform.get_feature_names()
    write_pickle(xform, out_p_in, name_in)
    return xform_data

def chi_fun(df_in, label_in, k_in, path_out, name_in):
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import SelectKBest
    import pandas as pd
    feat_sel = SelectKBest(score_func=chi2, k=k_in)
    dim_data = pd.DataFrame(feat_sel.fit_transform(
        df_in, label_in))
    feat_index = feat_sel.get_support(indices=True)
    feature_names = df_in.columns[feat_index]
    dim_data.columns = feature_names
    write_pickle(feat_sel, path_out, name_in)
    return dim_data

def jaccard_dist(corp_1, corp_2):
    #intersection/union
    intersection = set(corp_1.split()).intersection(set(corp_2.split()))
    union = set(corp_1.split()).union(set(corp_2.split()))
    return len(intersection) / len(union)

def cos_sim_fun(df_a, df_b, label_in):
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd
    import numpy as np
    cos_matrix = pd.DataFrame(cosine_similarity(
        df_a, df_b))
    cos_matrix.index = label_in
    cos_matrix.columns = label_in
    np.array(cos_matrix)
    print (np.average(np.array(cos_matrix)))
    return cos_matrix

def extract_embeddings_pre(df_in, out_path_i, name_in):
    #https://code.google.com/archive/p/word2vec/
    #https://pypi.org/project/gensim/
    #pip install gensim
    import pandas as pd
    from nltk.data import find
    from gensim.models import KeyedVectors
    import pickle
    def get_score(var):
        import numpy as np
        tmp_arr = list()
        for word in var:
            try:
                tmp_arr.append(list(my_model_t.get_vector(word)))
            except:
                pass
        tmp_arr
        return np.mean(np.array(tmp_arr), axis=0)
    word2vec_sample = str(find(name_in))
    my_model_t = KeyedVectors.load_word2vec_format(
        word2vec_sample, binary=False)
    # word_dict = my_model.key_to_index
    tmp_out = df_in.str.split().apply(get_score)
    tmp_data = tmp_out.apply(pd.Series).fillna(0)
    pickle.dump(my_model_t, open(out_path_i + "embeddings.pkl", "wb"))
    pickle.dump(tmp_data, open(out_path_i + "embeddings_df.pkl", "wb" ))
    return tmp_data, my_model_t

def domain_train(df_in, path_in, name_in):
    #domain specific
    import pandas as pd
    import gensim
    def get_score(var):
        import numpy as np
        tmp_arr = list()
        for word in var:
            try:
                tmp_arr.append(list(model.wv.get_vector(word)))
            except:
                pass
        tmp_arr
        return np.mean(np.array(tmp_arr), axis=0)
    model = gensim.models.Word2Vec(df_in.str.split())
    model.save(path_in + 'body.embedding')
    #call up the model
    #load_model = gensim.models.Word2Vec.load('body.embedding')
    model.wv.similarity('fish','river')
    tmp_data = pd.DataFrame(df_in.str.split().apply(get_score))
    return tmp_data, model

def pca_fun(df_in, exp_var_in, out_path_i, name_in):
    from sklearn.decomposition import PCA
    import pandas as pd
    my_pca = PCA(n_components=exp_var_in)
    pca_vec = pd.DataFrame(my_pca.fit_transform(df_in))
    exp_var = sum(my_pca.explained_variance_ratio_)
    print (exp_var)
    write_pickle(my_pca, out_path_i, name_in)
    return pca_vec

def model_train(df_in, label_in, test_size_in, sw_in, o_path):
    #sw = "gnb"
    from sklearn.model_selection import train_test_split
    import pandas as pd
    from sklearn.metrics import precision_recall_fscore_support
    X_train, X_test, y_train, y_test = train_test_split(
        df_in, label_in, test_size=test_size_in, random_state=42) #80/20
    if sw_in == "rf":
        from sklearn.ensemble import RandomForestClassifier
        my_model = RandomForestClassifier(random_state=123)
    elif sw_in == "svm":
        from sklearn.svm import SVC
        my_model = SVC()
    elif sw_in == "gnb":
        from sklearn.naive_bayes import GaussianNB
        my_model = GaussianNB()
    my_model.fit(X_train, y_train)
    write_pickle(my_model, o_path, sw_in)
    try:
        y_pred = pd.DataFrame(my_model.predict(X_test))
        #y_pred_proba = pd.DataFrame(my_model.predict_proba(X_test))
        #y_pred_proba.columns = my_model.classes_
    
        #performance
        perf = pd.DataFrame(precision_recall_fscore_support(
            y_test, y_pred, average='weighted'))
        perf.index = ["precision", "recall", "fscore", None]
        print (perf)
    except:
        print ("can't extract likelihood scores from", sw_in)
        pass
    return my_model

def grid_fun(df_in, label_in, grid_in, t_size_in, cv_in, o_path_in, sw_in):
    #gridsearchcv
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    
    X_train, X_test, y_train, y_test = train_test_split(
            df_in, label_in, test_size=t_size_in, random_state=42)
    if sw_in == "rf":
        grid_model = RandomForestClassifier()
    elif sw_in == "svm":
        grid_model = SVC()
    elif sw_in == "gnb":
        grid_model = GaussianNB()
    
    grid_search = GridSearchCV(
        estimator=grid_model, param_grid=grid_in, cv=cv_in)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print ("best params", best_params)
    if sw_in == "rf":
        grid_model = RandomForestClassifier(**grid_search.best_params_)
    elif sw_in == "svm":
        grid_model = SVC(**grid_search.best_params_)
    elif sw_in == "gnb":
        grid_model = GaussianNB(**grid_search.best_params_)
    grid_model.fit(df_in, label_in)
    write_pickle(grid_model, o_path_in, sw_in)
    return grid_model