Имеется множество новостей с одного новостного ресурса. Необходимо кластеризовать их по событиям, к которым они относятся. То есть отдельно должен быть кластер про пожар, отдельно про повышение пенсий и т.д.   
В наличии имеются и тексты новостей и тексты заголовков. Отдельно векторизуются tfidf-ом тексты новостей и заголовки, полученные матрицы объединяются в одну единую матрицу.   
Кластеризуем с помощью DBSCAN. Параметры:   
- min_samples = 1
- epsilon варьируется в зависимости от количества сообщений в наборе, но при значении epsilon = 0.5 почти всегда получаются +- сбалансированные кластеры.