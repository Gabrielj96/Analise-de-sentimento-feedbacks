def pred_clean(string):
    # bibliotecas padrões
    import pandas as pd
    import numpy as np

    # bibliotecas para NLP
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    import re
    from unidecode import unidecode
    from sklearn.feature_extraction.text import TfidfVectorizer

    # bibliotecas para modelagem
    from sklearn.svm import SVC
    from sklearn.metrics import precision_recall_fscore_support, classification_report
    from sklearn.model_selection import KFold, cross_validate
    from joblib import load

    # dados a classificar
    string_feedback = string
    string_feedback = {'Feedback': string_feedback}
    dados = pd.DataFrame({'Feedback': string_feedback})
    dados

    # limpeza dos dados
    dados.Feedback = dados.Feedback.str.lower()
    dados['Feedback'] = dados['Feedback'].apply(lambda x: unidecode(x))

    def RemovePunctuation(feedback):
        feedback = re.sub(r"[-|0-9]", "", feedback).lower()
        feedback = re.sub(r'[-./?!,":;()\']', ' ', feedback).lower()
        return (feedback)

    lista_feedback = []
    lista_feedback = [RemovePunctuation(feedback) for feedback in dados.Feedback]

    dados.Feedback = lista_feedback

    dados['Feedback'] = dados['Feedback'].apply(lambda x: x.strip())
    dados['Feedback'] = dados['Feedback'].apply(lambda x: x.replace("  ", " "))

    # Pŕe-Processamento
    nlp = word_tokenize

    dados['Contagem_palavras'] = dados['Feedback'].apply(lambda x: len(str(x).split(" ")))

    def RemoveStopWords(feedback):
        palavras = [i for i in feedback.split() if not i in corr_palavras]
        return (" ".join(palavras))

    lemmatizer = WordNetLemmatizer()
    def Lemmatization(feedback):
      palavras = []
      for w in feedback.split():
        palavras.append(lemmatizer.lemmatize(w))
      return (" ".join(palavras))

    #dados = dados[dados['Contagem_palavras'] > 2]

    corr_palavras = pd.read_csv('palavras_corr.csv')
    corr_palavras = list(corr_palavras.Palavras_corr.values)

    lista_feedback = []
    lista_feedback = [RemoveStopWords(feedback) for feedback in dados.Feedback]

    dados['Feedback_palavras_irrelevantes'] = lista_feedback

    # função para transformar o texto em Matrix Count e TFIDF
    tfidf = load('vectorizer_tfidf.joblib')

    def bow(feedbacks):
        bag_of_words_transformer = tfidf
        mx = bag_of_words_transformer.transform(feedbacks).todense()
        terms = bag_of_words_transformer.get_feature_names()
        dados_tfidf = pd.DataFrame(mx, columns=terms, index=feedbacks)

        return (dados_tfidf)

    feedback_lem = dados['Feedback_palavras_irrelevantes'].apply(lambda x: Lemmatization(x))
    dados_tfidf = bow(feedback_lem)

    svc = load('model_svc.joblib')

    classe = svc.predict(dados_tfidf)

    if classe in classe == 0:
        classe = 'Negativo'
    else:
        classe = 'Positivo'

    proba = svc.predict_proba(dados_tfidf)
    proba0 = proba[0][0]
    proba1 = proba[0][1]
    proba0 = str(round(proba0 *100, 2)) + " %"
    proba1 = str(round(proba1 *100, 2)) + " %"

    return (classe, proba0, proba1)
