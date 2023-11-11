import base64
import streamlit as st
import datetime
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor


@st.cache_data
def load_data(filename=None):
    filename_default = 'world_energy_consumption.csv'
    if not filename:
        filename = filename_default

    df = pd.read_csv(f"./{filename}")
    df = df[df.country != 'World']  # записей мирового показателя
    df = df[df.year > 2010]  # устранение записей старше 2010 года

    return df


# Вставка данных в БД
def add_in_db(my_mineral, my_model, my_country, my_production, my_mse, my_mae, my_r2, my_cross_val):
    conn, cursor = None, None
    try:
        conn = psycopg2.connect(dbname='postgres', user='postgres',
                                password='1234567890', host='localhost')
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        if my_mineral == 'oil':
            cursor.execute('''INSERT INTO oil (my_date, model_name, country_name, oil_prod,mse_val,mae_val,r2_val,cross_valid_val)
             VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s)''', (my_model, my_country, my_production, my_mse, my_mae, my_r2, my_cross_val))

        elif my_mineral == 'coal':
            cursor.execute('''INSERT INTO coal (my_date, model_name, country_name, coal_prod,mse_val,mae_val,r2_val,cross_valid_val)
                        VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s)''',
                           (my_model, my_country, my_production, my_mse, my_mae, my_r2, my_cross_val))
        elif my_mineral == 'gas':
            cursor.execute('''INSERT INTO gas (my_date, model_name, country_name, gas_prod,mse_val, mae_val, r2_val, cross_valid_val)
                        VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s)''',
                           (my_model, my_country, my_production, my_mse, my_mae, my_r2, my_cross_val))
    except Exception as e:
        print(e)
    finally:
        if conn:
            cursor.close()
            conn.close()
            print("Соединение с базой данных закрыто")


# Метод фильтрации данных под категорию нефть
@st.cache_data
def oil_data(df):
    oil_cols = [col for col in df.columns if 'oil' in col]
    # общие показатели для всех полезных ископаемых
    oil_cols.extend(['gdp', 'population', 'year', 'country'])
    oil_cols.reverse()  # разворот датасета для отображения правильной последовательности колонок

    # формирование датасета нефтяной промышленности
    oil_data = df[oil_cols]
    # очистка данных на душу населения (они усложняют процесс обучения не играют роли)
    oil_data = oil_data.drop(
        ['oil_prod_per_capita', 'oil_prod_change_twh', 'oil_prod_change_pct', 'oil_elec_per_capita',
         'oil_energy_per_capita'], axis=1)
    # ликвидация пропусков
    oil_data = oil_data.dropna()

    return oil_data


# Метод фильтрации данных под категорию газ
@st.cache_data
def gas_data(df):
    gas_cols = [col for col in df.columns if ('gas' in col) and (
            col != 'gas_prod_per_capita' and col != 'gas_prod_change_twh')]  # сортировка по показателям газа
    gas_cols.extend(['gdp', 'population', 'year', 'country'])  # общие показатели для всех полезных ископаемых
    gas_cols.reverse()  # разворот датасета для отображения правильной последовательности колонок

    # формирование датасета газовой промышленности
    gas_data = df[gas_cols]
    # очистка данных на душу населения (они усложняют процесс обучения не играют роли)
    gas_data = gas_data.drop(['gas_elec_per_capita', 'gas_energy_per_capita', 'gas_prod_change_pct'], axis=1)
    # ликвидация пропусков
    gas_data = gas_data.dropna()

    return gas_data


# Метод фильтрации данных под категорию уголь
@st.cache_data
def coal_data(df):
    coal_cols = [col for col in df.columns if ('coal' in col) and (
            col != 'coal_prod_per_capita' and col != 'coal_prod_change_twh')]  # сортировка по показателям угля
    coal_cols.extend(['gdp', 'population', 'year', 'country'])  # общие показатели для всех полезных ископаемых
    coal_cols.reverse()  # разворот датасета для отображения правильной последовательности колонок

    # формирование датасета промышленности угля
    coal_data = df[coal_cols]
    # очистка данных на душу населения (они усложняют процесс обучения не играют роли)
    coal_data = coal_data.drop(['coal_elec_per_capita', 'coal_cons_per_capita', 'coal_prod_change_pct'], axis=1)
    # ликвидация пропусков
    coal_data = coal_data.dropna()

    return coal_data


def add_logo(logo_path=None):
    if not logo_path:
        logo_path = open("logo.svg").read()
    b64 = base64.b64encode(logo_path.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    return html

# Модель машинного обучения - Дерево решений (с GridSearch)
@st.cache_resource
def decision_tree_model(data, mineral, user_data=None, check=False):
    model_name = 'DecisionTreeReg'
    X, y = None, None
    enc = LabelEncoder()  # кодировщик категориальных данных
    data['country'] = enc.fit_transform(data['country'])
    if check:  # если пользовательские данные
        user_data['country'] = enc.fit_transform(user_data['country'])
    else:
        pass
    if mineral == 'oil':
        # Разделение датасетов на обучающую и тестовую выборки
        X = data.drop(['oil_production'], axis=1)
        y = data[['oil_production']]
    elif mineral == 'coal':
        # Разделение датасетов на обучающую и тестовую выборки
        X = data.drop(['coal_production'], axis=1)
        y = data[['coal_production']]
    elif mineral == 'gas':
        # Разделение датасетов на обучающую и тестовую выборки
        X = data.drop(['gas_production'], axis=1)
        y = data[['gas_production']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # параметры модели
    param_grid = {
        'max_depth': [3, 4, 5, 6, 7, 8, 9],  # глубина дерева
        'max_features': ['sqrt', 'log2'],  # количество функций, которые необходимо учитывать при поиске лучшего сплита
        'min_samples_split': [2, 5, 10],  # минимальное количество образцов, требуемых для разделения узла
        'min_samples_leaf': [1, 2, 4],  # минимальное количество образцов, требуемых в листовом узле
        'ccp_alpha': [0.1, 1, 10],
        'criterion': ['friedman_mse', 'absolute_error', 'poisson', 'squared_error']   # определяет критерий, используемый для измерения качества разделения
    }

    # Модель
    model_tree = DecisionTreeRegressor(random_state=42)
    # поиск лучших параметров через gridsearch
    grid_search = GridSearchCV(model_tree, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    start_time = datetime.datetime.now()  # время начала запуска
    cv1 = KFold(n_splits=10, shuffle=True)  # 10 разбиений и случайное перемешивание данных на каждой итерации
    cv_score = cross_val_score(grid_search, X, y, scoring="r2", cv=cv1).mean()   # перекрест проверка + среднее значение
    grid_search.fit(X_train, y_train)  # обучение
    best_regressor = grid_search.best_estimator_  # наилучшая комбинация в модели
    if check:  # если работаем с пользовательскими данными
        y_pred_user_tree = best_regressor.predict(user_data)
        if mineral == 'oil':
            st.markdown(f':green[**Предсказанный объём добычи нефти:**] {y_pred_user_tree[0]}')
        elif mineral == 'gas':
            st.markdown(f':green[**Предсказанный объём добычи газа:**] {y_pred_user_tree[0]}')
        elif mineral == 'coal':
            st.markdown(f':green[**Предсказанный объём добычи угля:**] {y_pred_user_tree[0]}')
    else:  # если работаем с базовыми данными
        # Оценка модели с оптимальными параметрами
        y_pred_tree = best_regressor.predict(X_test)  # прогнозы
        show_predictions(y_test, y_pred_tree, cv_score,
                         enc.inverse_transform(data['country'].iloc[len(X_train):len(data['country'])]),
                         mineral, model_name)
    st.write(f'Runtime: {(datetime.datetime.now() - start_time).seconds} s')
    st.info("[Справка по модели DecisionTree](https://www.almabetter.com/bytes/tutorials/data-science/decision-tree)")


# Модель машинного обучения - Бэггинг-регрессия
@st.cache_resource
def bagging_reg_model(data, mineral, user_data=None, check=False):
    model_name = 'BaggingReg'
    X, y = None, None
    enc = LabelEncoder()  # кодировщик категориальных данных
    data['country'] = enc.fit_transform(data['country'])
    if check:  # если пользовательские данные
        user_data['country'] = enc.fit_transform(user_data['country'])
    else:
        pass
    if mineral == 'oil':
        # Разделение датасетов на обучающую и тестовую выборки
        X = data.drop(['oil_production'], axis=1)
        y = data[['oil_production']]
    elif mineral == 'coal':
        # Разделение датасетов на обучающую и тестовую выборки
        X = data.drop(['coal_production'], axis=1)
        y = data[['coal_production']]
    elif mineral == 'gas':
        # Разделение датасетов на обучающую и тестовую выборки
        X = data.drop(['gas_production'], axis=1)
        y = data[['gas_production']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # Базовая модель
    dtree = DecisionTreeRegressor(random_state=0)
    # Модель бэггинга
    br = BaggingRegressor(estimator=dtree, n_estimators=100, bootstrap=True, random_state=0)
    start_time = datetime.datetime.now()  # время начала запуска
    cv1 = KFold(n_splits=10, shuffle=True)  # 10 разбиений и случайное перемешивание данных на каждой итерации
    cv_score = cross_val_score(br, X, y, scoring="r2", cv=cv1).mean()  # перекрест проверка + среднее значение
    # Обучение
    br.fit(X_train, y_train.values.ravel())
    # Прогнозы
    if check:  # если работаем с пользовательскими данными
        y_pred_user_br = br.predict(user_data)
        if mineral == 'oil':
            st.markdown(f':green[**Предсказанный объём добычи нефти:**] {y_pred_user_br[0]}')
        elif mineral == 'gas':
            st.markdown(f':green[**Предсказанный объём добычи газа:**] {y_pred_user_br[0]}')
        elif mineral == 'coal':
            st.markdown(f':green[**Предсказанный объём добычи угля:**] {y_pred_user_br[0]}')
    else:  # если работаем с базовыми данными
        y_pred_br = br.predict(X_test)

        show_predictions(y_test, y_pred_br, cv_score,
                         enc.inverse_transform(data['country'].iloc[len(X_train):len(data['country'])]),
                         mineral, model_name)
    st.write(f'Runtime: {(datetime.datetime.now() - start_time).seconds} s')
    st.info("[Справка по модели Bagging](https://www.simplilearn.com/tutorials/machine-learning-tutorial/bagging-in-machine-learning)")


# Модель машинного обучения - Ада-Бустер-регрессия
@st.cache_resource
def ada_boost_model(data, mineral, user_data=None, check=False):
    model_name = 'AdaBoost'
    X, y = None, None
    enc = LabelEncoder()  # кодировщик категориальных данных
    data['country'] = enc.fit_transform(data['country'])
    if check:  # если пользовательские данные
        user_data['country'] = enc.fit_transform(user_data['country'])
    else:
        pass

    if mineral == 'oil':
        # Разделение датасетов на обучающую и тестовую выборки
        X = data.drop(['oil_production'], axis=1)
        y = data[['oil_production']]
    elif mineral == 'coal':
        # Разделение датасетов на обучающую и тестовую выборки
        X = data.drop(['coal_production'], axis=1)
        y = data[['coal_production']]
    elif mineral == 'gas':
        # Разделение датасетов на обучающую и тестовую выборки
        X = data.drop(['gas_production'], axis=1)
        y = data[['gas_production']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # Базовая модель
    dtree = DecisionTreeRegressor(random_state=0)
    # Модель бустера
    adb = AdaBoostRegressor(estimator=dtree, n_estimators=100, random_state=0)
    start_time = datetime.datetime.now()  # время начала запуска
    cv1 = KFold(n_splits=10, shuffle=True)  # 10 разбиений и случайное перемешивание данных на каждой итерации
    cv_score = cross_val_score(adb, X, y, scoring="r2", cv=cv1).mean()  # перекрест проверка + среднее значение
    # Обучение
    adb.fit(X_train, y_train.values.ravel())
    # Прогнозы
    if check:  # если работаем с пользовательскими данными
        y_pred_user_adb = adb.predict(user_data)
        if mineral == 'oil':
            st.markdown(f':green[**Предсказанный объём добычи нефти:**] {y_pred_user_adb[0]}')
        elif mineral == 'gas':
            st.markdown(f':green[**Предсказанный объём добычи газа:**] {y_pred_user_adb[0]}')
        elif mineral == 'coal':
            st.markdown(f':green[**Предсказанный объём добычи угля:**] {y_pred_user_adb[0]}')
    else:  # если работаем с базовыми данными
        y_pred_adb = adb.predict(X_test)
        show_predictions(y_test, y_pred_adb, cv_score,
                         enc.inverse_transform(data['country'].iloc[len(X_train):len(data['country'])]),
                         mineral, model_name)
    st.write(f'Runtime: {(datetime.datetime.now() - start_time).seconds} s')
    st.info("[Справка по модели AdaBoost](https://www.almabetter.com/bytes/tutorials/data-science/adaboost-algorithm)")


# Модель машинного обучения - Случайный-лес-регрессия (Бэггинг)
@st.cache_resource
def random_forest_model(data, mineral, user_data=None, check=False):
    model_name = 'RandomForest'
    X, y = None, None
    enc = LabelEncoder()  # кодировщик категориальных данных
    data['country'] = enc.fit_transform(data['country'])
    if check:  # если пользовательские данные
        user_data['country'] = enc.fit_transform(user_data['country'])
    else:
        pass
    if mineral == 'oil':
        # Разделение датасетов на обучающую и тестовую выборки
        X = data.drop(['oil_production'], axis=1)
        y = data[['oil_production']]
    elif mineral == 'coal':
        # Разделение датасетов на обучающую и тестовую выборки
        X = data.drop(['coal_production'], axis=1)
        y = data[['coal_production']]
    elif mineral == 'gas':
        # Разделение датасетов на обучающую и тестовую выборки
        X = data.drop(['gas_production'], axis=1)
        y = data[['gas_production']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Создание модели Случайного леса (число деревьев = 100, n_jobs=-1 использование всех доступных ядер процессора, random_state = 0 - одни и те же значения)
    rfr = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0)
    # Обучение
    start_time = datetime.datetime.now()  # время начала запуска
    cv1 = KFold(n_splits=10, shuffle=True)  # 10 разбиений и случайное перемешивание данных на каждой итерации
    cv_score = cross_val_score(rfr, X, y, scoring="r2", cv=cv1).mean()  # перекрест проверка + среднее значение
    rfr.fit(X_train, y_train.values.ravel())
    # Прогнозы модели
    if check:  # если работаем с пользовательскими данными
        y_pred_user_rfr = rfr.predict(user_data)
        if mineral == 'oil':
            st.markdown(f':green[**Предсказанный объём добычи нефти:**] {y_pred_user_rfr[0]}')
        elif mineral == 'gas':
            st.markdown(f':green[**Предсказанный объём добычи газа:**] {y_pred_user_rfr[0]}')
        elif mineral == 'coal':
            st.markdown(f':green[**Предсказанный объём добычи угля:**] {y_pred_user_rfr[0]}')
    else:  # если работаем с базовыми данными
        y_pred_rfr = rfr.predict(X_test)
        show_predictions(y_test, y_pred_rfr, cv_score,
                         enc.inverse_transform(data['country'].iloc[len(X_train):len(data['country'])]),
                         mineral, model_name)
    st.write(f'Runtime: {(datetime.datetime.now() - start_time).seconds} s')
    st.info("[Справка по модели RandomForest](https://www.almabetter.com/bytes/tutorials/data-science/random-forest)")


# Модель машинного обучения - Градиент-бустер-регрессия
@st.cache_resource
def gradient_boost_model(data, mineral, user_data=None, check=False):
    model_name = 'GradientBoost'
    X, y = None, None
    enc = LabelEncoder()  # кодировщик категориальных данных
    data['country'] = enc.fit_transform(data['country'])
    if check:  # если пользовательские данные
        user_data['country'] = enc.fit_transform(user_data['country'])
    else:
        pass
    if mineral == 'oil':
        # Разделение датасетов на обучающую и тестовую выборки
        X = data.drop(['oil_production'], axis=1)
        y = data[['oil_production']]
    elif mineral == 'coal':
        # Разделение датасетов на обучающую и тестовую выборки
        X = data.drop(['coal_production'], axis=1)
        y = data[['coal_production']]
    elif mineral == 'gas':
        # Разделение датасетов на обучающую и тестовую выборки
        X = data.drop(['gas_production'], axis=1)
        y = data[['gas_production']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # Модель бустинга
    grb = GradientBoostingRegressor(n_estimators=100, random_state=0)
    # Обучение
    start_time = datetime.datetime.now()  # время начала запуска
    cv1 = KFold(n_splits=10, shuffle=True)  # 10 разбиений и случайное перемешивание данных на каждой итерации
    cv_score = cross_val_score(grb, X, y, scoring="r2", cv=cv1).mean()  # перекрест проверка + среднее значение
    grb.fit(X_train, y_train.values.ravel())
    # Прогнозы
    if check:  # если работаем с пользовательскими данными
        y_pred_user_grb = grb.predict(user_data)
        if mineral == 'oil':
            st.markdown(f':green[**Предсказанный объём добычи нефти:**] {y_pred_user_grb[0]}')
        elif mineral == 'gas':
            st.markdown(f':green[**Предсказанный объём добычи газа:**] {y_pred_user_grb[0]}')
        elif mineral == 'coal':
            st.markdown(f':green[**Предсказанный объём добычи угля:**] {y_pred_user_grb[0]}')
    else:  # если работаем с базовыми данными
        y_pred_grb = grb.predict(X_test)
        show_predictions(y_test, y_pred_grb, cv_score,
                         enc.inverse_transform(data['country'].iloc[len(X_train):len(data['country'])]),
                         mineral, model_name)
    st.write(f'Runtime: {(datetime.datetime.now() - start_time).seconds} s')
    st.info("[Справка по модели GradientBoost](https://www.almabetter.com/bytes/tutorials/data-science/gradient-boosting)")


# Метод отображения результатов работы модели, метрик оценок качества
def show_predictions(y_test, y_pred, cv_score, country, mineral, model_name):
    # Формирование сета для оценки результатов
    compare_lin = pd.DataFrame(columns=['Результат', 'Прогноз'])
    if mineral == 'oil':  # если категория - нефть
        for pair in list(zip(y_test['oil_production'], y_pred)):
            compare_lin.loc[len(compare_lin.index)] = [pair[0], pair[1]]
        compare_lin['MSE'] = mean_squared_error(y_test['oil_production'], y_pred)
        compare_lin['MAE'] = mean_absolute_error(y_test['oil_production'], y_pred)
        compare_lin['R^2'] = r2_score(y_test['oil_production'], y_pred)
        compare_lin['Cross_val_score'] = cv_score
        compare_lin['Country'] = country
        compare_lin = compare_lin.sort_values(by=['Прогноз'], ascending=False)
        # Вывод
        st.write('Рекомендуемая страна для добычи нефти:')
        st.subheader(f'{compare_lin["Country"].iloc[0]}')
        st.write('')
        st.markdown(f'\n :green[**Предсказанный объём добычи нефти:**] {compare_lin["Прогноз"].iloc[0]:.2f}')
        st.markdown(f'\n :orange[MSE (Mean Squared Error):] {compare_lin["MSE"].iloc[0]:.2f}')
        st.markdown(f'\n :orange[MAE (Mean Absolute Error):] {compare_lin["MAE"].iloc[0]:.2f}')
        st.markdown(f'\n :orange[R^2 score:] {compare_lin["R^2"].iloc[0]:.2f}')
        st.markdown(f'\n :orange[Cross validation score:] {compare_lin["Cross_val_score"].iloc[0]:.2f}')

        add_in_db(mineral, model_name, compare_lin["Country"].iloc[0], compare_lin["Прогноз"].iloc[0],
                  compare_lin["MSE"].iloc[0], compare_lin["MAE"].iloc[0], compare_lin["R^2"].iloc[0], compare_lin["Cross_val_score"].iloc[0])
        st.write(compare_lin.head())
    elif mineral == 'coal':  # если категория - уголь
        for pair3 in list(zip(y_test['coal_production'], y_pred)):
            compare_lin.loc[len(compare_lin.index)] = [pair3[0], pair3[1]]
        compare_lin['MSE'] = mean_squared_error(y_test['coal_production'], y_pred)
        compare_lin['MAE'] = mean_absolute_error(y_test['coal_production'], y_pred)
        compare_lin['R^2'] = r2_score(y_test['coal_production'], y_pred)
        compare_lin['Cross_val_score'] = cv_score
        compare_lin['Country'] = country
        compare_lin.insert(0, "Mineral_resource", "Coal")
        compare_lin = compare_lin.sort_values(by=['Прогноз'], ascending=False)
        # Вывод
        st.write('Рекомендуемая страна для добычи угля:')
        st.subheader(f'{compare_lin["Country"].iloc[0]}')
        st.write('')
        st.markdown(f'\n :green[**Предсказанный объём добычи угля:**] {compare_lin["Прогноз"].iloc[0]:.2f}')
        st.markdown(f'\n :orange[MSE (Mean Squared Error):] {compare_lin["MSE"].iloc[0]:.2f}')
        st.markdown(f'\n :orange[MAE (Mean Absolute Error):] {compare_lin["MAE"].iloc[0]:.2f}')
        st.markdown(f'\n :orange[R^2 score:] {compare_lin["R^2"].iloc[0]:.2f}')
        st.markdown(f'\n :orange[Cross validation score:] {compare_lin["Cross_val_score"].iloc[0]:.2f}')

        add_in_db(mineral, model_name, compare_lin["Country"].iloc[0], compare_lin["Прогноз"].iloc[0],
                  compare_lin["MSE"].iloc[0], compare_lin["MAE"].iloc[0], compare_lin["R^2"].iloc[0], compare_lin["Cross_val_score"].iloc[0])
        st.write(compare_lin.head())
    elif mineral == 'gas':  # если категория - газ
        for pair2 in list(zip(y_test['gas_production'], y_pred)):
            compare_lin.loc[len(compare_lin.index)] = [pair2[0], pair2[1]]
        compare_lin['MSE'] = mean_squared_error(y_test['gas_production'], y_pred)
        compare_lin['MAE'] = mean_absolute_error(y_test['gas_production'], y_pred)
        compare_lin['R^2'] = r2_score(y_test['gas_production'], y_pred)
        compare_lin['Cross_val_score'] = cv_score
        compare_lin['Country'] = country
        compare_lin.insert(0, "Mineral_resource", "Gas")
        compare_lin = compare_lin.sort_values(by=['Прогноз'], ascending=False)
        # Вывод
        st.write('Рекомендуемая страна для добычи газа:')
        st.subheader(f'{compare_lin["Country"].iloc[0]}')
        st.write('')
        st.markdown(f'\n :green[**Предсказанный объём добычи газа:**] {compare_lin["Прогноз"].iloc[0]:.2f}')
        st.markdown(f'\n :orange[MSE (Mean Squared Error):] {compare_lin["MSE"].iloc[0]:.2f}')
        st.markdown(f'\n :orange[MAE (Mean Absolute Error):] {compare_lin["MAE"].iloc[0]:.2f}')
        st.markdown(f'\n :orange[R^2 score:] {compare_lin["R^2"].iloc[0]:.2f}')
        st.markdown(f'\n :orange[Cross validation score:] {compare_lin["Cross_val_score"].iloc[0]:.2f}')

        add_in_db(mineral, model_name, compare_lin["Country"].iloc[0], compare_lin["Прогноз"].iloc[0],
                  compare_lin["MSE"].iloc[0], compare_lin["MAE"].iloc[0], compare_lin["R^2"].iloc[0], compare_lin["Cross_val_score"].iloc[0])
        st.write(compare_lin.head())

    # группировка по странам и сортировка по спрогнозированному показателю объёма добычи
    data_models_oil_grouped = compare_lin.groupby(['Country']).first().reset_index().sort_values(by=['Прогноз'], ascending=True)
    x = data_models_oil_grouped['Country'].tolist()
    y = data_models_oil_grouped['Прогноз'].tolist()

    fig, ax = plt.subplots(figsize=(20, 8))
    bars = None # столбчатые графики
    plt.title('Объём добычи нефти')
    plt.xlabel('ТВтч')

    if mineral == 'oil':
        bars = plt.barh(x, y, color='r', height=0.7)
    elif mineral == 'coal':
        bars = plt.barh(x, y, color='g', height=0.7)
    elif mineral == 'gas':
        bars = plt.barh(x, y, color='b', height=0.7)

    # отображение значений на графиках
    ax.bar_label(bars, padding=-45, color='black',
                 fontsize=14, label_type='edge', fmt='%.1f',
                 fontweight='bold')
    st.pyplot(fig)


# Метод главной страницы
def home_page_builder():
    st.title("Добыча полезных ископаемых")
    st.divider()
    st.markdown('''**Веб-приложение с использованием искусственного интеллекта способно спрогнозировать
             производственные результаты добычи полезных ископаемых. Оно собирает данные и анализирует их,
              чтобы предсказать объёмы добычи и оптимальные места для бурения скважин.**''')
    col1, col2, col3 = st.columns(3)

    # Отображение картинок категорий полезных ископаемых
    with col1:
        image_oil = Image.open('oil.png')
        st.header("Нефть")
        st.image(image_oil)

    with col2:
        image_coal = Image.open('coal.png')
        st.header("Уголь")
        st.image(image_coal)

    with col3:
        image_gas = Image.open('gas.png')
        st.header("Газ")
        st.image(image_gas)
    st.write('')
    st.write('')


# Метод работы с пользовательскими данными для прогнозирования
def user_data(mineral):
    user_predata = None
    st.title('**Пользовательские данные**')
    st.divider()
    st.markdown(':gray[Заполните все поля соответствующими данными и нажмите кнопку START]')
    st.divider()
    country = st.text_input("Введите страну")
    year = st.number_input("Введите год", value=None, placeholder="Число")
    population = st.number_input("Введите население", value=None, placeholder="Число")
    gdp = st.number_input("Введите ВВП", value=None, placeholder="Число")

    if mineral == 'oil':  # если категория - нефть
        oil_consumption = st.number_input("Введите первичное потребление энергии из нефти, измеряемое в ТВтч",
                                          value=None, placeholder="Число")
        oil_cons_change_twh = st.number_input("Введите годовое изменение потребления нефти, измеряемое в ТВтч",
                                              value=None, placeholder="Число")
        oil_share_energy = st.number_input("Введите долю потребления первичной энергии, получаемая из нефти",
                                           value=None, placeholder="Число")
        oil_cons_change_pct = st.number_input("Введите ежегодное процентное изменение потребления масла",
                                              value=None, placeholder="Число")
        oil_share_elec = st.number_input("Введите долю потребления электроэнергии, получаемая из нефти",
                                         value=None, placeholder="Число")
        oil_electricity = st.number_input("Введите производство электроэнергии из нефти, измеряется в ТВтч",
                                          value=None, placeholder="Число")
        user_predata = {'country': [country],
                        'year': [year],
                        'population': [population],
                        'gdp': [gdp],
                        'oil_consumption': [oil_consumption],
                        'oil_cons_change_twh': [oil_cons_change_twh],
                        'oil_share_energy': [oil_share_energy],
                        'oil_cons_change_pct': [oil_cons_change_pct],
                        'oil_share_elec': [oil_share_elec],
                        'oil_electricity': [oil_electricity]

                        }
    elif mineral == 'coal':   # если категория - уголь
        coal_electricity = st.number_input("Введите производство электроэнергии из угля, измеряется в ТВтч",
                                           value=None, placeholder="Число")
        coal_consumption = st.number_input("Введите первичное потребление энергии из газа, измеряемое в ТВтч",
                                           value=None, placeholder="Число")
        coal_cons_change_twh = st.number_input("Введите годовое изменение потребления угля, измеряемое в ТВтч",
                                               value=None, placeholder="Число")
        coal_share_energy = st.number_input("Введите долю потребления первичной энергии, получаемая из угля",
                                            value=None, placeholder="Число")
        coal_cons_change_pct = st.number_input("Введите ежегодное процентное изменение потребления угля",
                                               value=None, placeholder="Число")
        coal_share_elec = st.number_input("Доля потребление электроэнергии, получаемая из угля",
                                          value=None, placeholder="Число")
        user_predata = {'country': [country],
                        'year': [year],
                        'population': [population],
                        'gdp': [gdp],
                        'coal_electricity': [coal_electricity],
                        'coal_consumption': [coal_consumption],
                        'coal_cons_change_twh': [coal_cons_change_twh],
                        'coal_share_energy': [coal_share_energy],
                        'coal_cons_change_pct': [coal_cons_change_pct],
                        'coal_share_elec': [coal_share_elec]

                        }
    elif mineral == 'gas':  # если категория - газ

        gas_electricity = st.number_input("Введите производство электроэнергии из газа, измеряется в ТВтч",
                                          value=None, placeholder="Число")
        gas_cons_change_twh = st.number_input("Введите годовое изменение потребления газа, измеряемое в ТВтч",
                                              value=None, placeholder="Число")
        gas_share_energy = st.number_input("Введите долю потребления первичной энергии, получаемая из газа",
                                           value=None, placeholder="Число")
        gas_cons_change_pct = st.number_input("Введите ежегодное процентное изменение потребления газа",
                                              value=None, placeholder="Число")
        gas_share_elec = st.number_input("Введите долю потребления электроэнергии, получаемая из газа",
                                         value=None, placeholder="Число")
        gas_consumption = st.number_input("Введите первичное потребление энергии из газа, измеряемое в ТВтч",
                                          value=None, placeholder="Число")
        user_predata = {'country': [country],
                        'year': [year],
                        'population': [population],
                        'gdp': [gdp],
                        'gas_consumption': [gas_consumption],
                        'gas_cons_change_twh': [gas_cons_change_twh],
                        'gas_share_energy': [gas_share_energy],
                        'gas_cons_change_pct': [gas_cons_change_pct],
                        'gas_share_elec': [gas_share_elec],
                        'gas_electricity': [gas_electricity],
                        }
    user_df = pd.DataFrame(user_predata)

    return user_df


def main():
    df = load_data()

    oil_dataset = oil_data(df)
    # отбор данных для категории уголь
    coal_dataset = coal_data(df)
    # отбор данных для категории газ
    gas_dataset = gas_data(df)

    my_logo = add_logo()
    st.sidebar.write(my_logo, unsafe_allow_html=True)
    st.sidebar.title('Меню')
    # отображение данных полезных ископаемых
    check_box_data = st.sidebar.checkbox('Данные полезных ископаемых')
    # отображение графиков показателя "production" полезных ископаемых
    check_box_visual = st.sidebar.checkbox('Визуализация данных')

    if check_box_data:
        st.info(
            "[Датасет \"World Energy Consumption\"](https://www.kaggle.com/datasets/pralabhpoudel/world-energy-consumption)"
            ''' - этот набор данных представляет собой коллекцию ключевых показателей,
                     поддерживаемых компанией Our World in Data. Он регулярно обновляется и включает данные о
                      потреблении энергии (первичной энергии, на душу населения и темпы роста), структуре энергопотребления,
                       структуре электропотребления и другие важные показатели.'''
        )

        st.subheader(':red[Данные нефтяной промышленности]')
        oil_dataset = oil_data(df)
        st.write(
            f'Таблица-нефть содержит **{len(oil_dataset.axes[0])}** строк и **{len(oil_dataset.axes[1])}** колонок')
        st.write(oil_dataset.head())
        st.markdown(
            '1. oil_electricity |	Производство электроэнергии из нефти, измеряется в тераватт-часах\n2. oil_share_elec |	Доля потребления электроэнергии, получаемая из нефти\n3. oil_cons_change_pct |	Ежегодное процентное изменение потребления масла\n4. oil_share_energy |	Доля потребления первичной энергии, получаемая из нефти\n5. oil_cons_change_twh |	Годовое изменение потребления нефти, измеряемое в тераватт-часах\n6. oil_consumption |	Первичное потребление энергии из нефти, измеряемое в тераватт-часах\n7. oil_production |	Добыча нефти, измеряемая в тераватт-часах (*1 Тераватт-час эквивалентно 85 984,52 Тонны нефтяного эквивалента')

        st.subheader(':green[Данные угольной промышленности]')
        coal_dataset = coal_data(df)
        st.write(
            f'Таблица-уголь содержит **{len(coal_dataset.axes[0])}** строк и **{len(coal_dataset.axes[1])}** колонок')
        st.write(coal_dataset.head())
        st.write(
            '1. coal_electricity | Производство электроэнергии из угля, измеряется в тераватт-часах\n2. coal_share_elec | Доля потребления электроэнергии, получаемая из угля\n3. coal_cons_change_pct | Ежегодное процентное изменение потребления угля\n4. coal_share_energy | Доля потребления первичной энергии, получаемая из угля\n5. coal_cons_change_twh | Годовое изменение потребления угля, измеряемое в тераватт-часах\n6. coal_consumption | Первичное потребление энергии из угля, измеряемое в тераватт-часах угля на душу населения, измеряемое в киловатт-часах\n7. coal_production | Добыча угля, измеряемая в тераватт-часах')
        st.subheader(':blue[Данные газовой промышленности]')
        gas_dataset = gas_data(df)
        st.write(
            f'Таблица-газ содержит **{len(gas_dataset.axes[0])}** строк и **{len(gas_dataset.axes[1])}** колонок')
        st.write(gas_dataset.head())
        st.write(
            '1. gas_electricity |	Производство электроэнергии из газа, измеряется в тераватт-часах\n2. gas_share_elec |	Доля потребления электроэнергии, получаемая из газа\n3. gas_cons_change_pct|	Ежегодное процентное изменение потребления газа\n4. gas_share_energy |	Доля потребления первичной энергии, получаемая из газа\n5. gas_cons_change_twh |	Годовое изменение потребления газа, измеряемое в тераватт-часах\n6. gas_consumption | Первичное потребление энергии из газа, измеряемое в тераватт-часах\n7. gas_production |	Добыча газа, измеряемая в тераватт-часах\n')

    if check_box_visual:
        st.subheader('Данные нефтяной промышленности')
        oil_dataset = oil_data(df)
        oil_d2 = oil_dataset.sort_values(by=['oil_production'],
                                         ascending=True)  # сортировка значений добычи по убыванию
        fig = go.Figure(go.Bar(
            x=oil_d2['oil_production'],
            y=oil_d2['country'], marker=dict(color='rgba(255, 0, 0, 0.6)'),
            orientation='h'))
        st.plotly_chart(fig)

        st.subheader('Данные угольной промышленности')
        gas_dataset = gas_data(df)
        gas_d2 = gas_dataset.sort_values(by=['gas_production'],
                                         ascending=True)  # сортировка значений добычи по убыванию
        fig2 = go.Figure(go.Bar(
            x=gas_d2['gas_production'],
            y=gas_d2['country'], marker=dict(color='rgba(0, 184, 6, 0.6)'),
            orientation='h'))
        st.plotly_chart(fig2)

        st.subheader('Данные газовой промышленности')
        coal_dataset = coal_data(df)
        coal_d2 = coal_dataset.sort_values(by=['coal_production'],
                                           ascending=True)  # сортировка значений добычи по убыванию
        fig3 = go.Figure(go.Bar(
            x=coal_d2['coal_production'],
            y=coal_d2['country'], marker=dict(color='rgba(0, 102, 255, 0.6)'),
            orientation='h'))
        st.plotly_chart(fig3)

    choose_mineral = st.sidebar.selectbox("Выберите полезное ископаемое", [
        "Главная", "Нефть", "Уголь", "Газ"])

    if choose_mineral == "Главная":
        home_page_builder()

    elif choose_mineral == "Нефть":
        choose_model = st.sidebar.selectbox("Выберите модель машинного обучения", [
            "No model", "AdaBoost", "RandomForest", "GradientBoost", "BaggingRegressor", "DecisionTreeRegressor"])

        st.title(':red[Нефтяная промышленность]')
        st.divider()
        st.markdown(''':gray[Отрасль экономики, занимающаяся добычей, переработкой, транспортировкой,
        складированием и продажей полезного природного ископаемого — нефти и сопутствующих нефтепродуктов.]''')
        st.divider()

        if choose_model == 'GradientBoost':
            gradient_boost_model(oil_dataset, 'oil')
            # возможность ввода пользовательских данных
            check_box_user_data = st.sidebar.checkbox('Ввести свои данные')
            if check_box_user_data:
                user_df = user_data('oil')
                if st.button('START'):
                    gradient_boost_model(oil_dataset, 'oil', user_data=user_df, check=True)
        elif choose_model == 'RandomForest':
            random_forest_model(oil_dataset, 'oil')
            check_box_user_data = st.sidebar.checkbox('Ввести свои данные')
            if check_box_user_data:
                user_df = user_data('oil')
                if st.button('START'):
                    random_forest_model(oil_dataset, 'oil', user_data=user_df, check=True)
        elif choose_model == 'AdaBoost':
            ada_boost_model(oil_dataset, 'oil')
            check_box_user_data = st.sidebar.checkbox('Ввести свои данные')
            if check_box_user_data:
                user_df = user_data('oil')
                if st.button('START'):
                    ada_boost_model(oil_dataset, 'oil', user_data=user_df, check=True)
        elif choose_model == 'BaggingRegressor':
            bagging_reg_model(oil_dataset, 'oil')
            check_box_user_data = st.sidebar.checkbox('Ввести свои данные')
            if check_box_user_data:
                user_df = user_data('oil')
                if st.button('START'):
                    bagging_reg_model(oil_dataset, 'oil', user_data=user_df, check=True)
                else:
                    pass
        elif choose_model == 'DecisionTreeRegressor':
            decision_tree_model(oil_dataset, 'oil')
            check_box_user_data = st.sidebar.checkbox('Ввести свои данные')
            if check_box_user_data:
                user_df = user_data('oil')
                if st.button('START'):
                    decision_tree_model(oil_dataset, 'oil', user_data=user_df, check=True)
                else:
                    pass

    elif choose_mineral == "Уголь":
        choose_model = st.sidebar.selectbox("Выберите модель машинного обучения", [
            "No model", "AdaBoost", "RandomForest", "GradientBoost", "BaggingRegressor", "DecisionTreeRegressor"])

        st.title(':green[Угольная промышленность]')
        st.divider()
        st.markdown(''':gray[Отрасль топливной промышленности, которая включает добычу открытым способом или в шахтах,
         обогащение и переработку (брикетирование) бурого и каменного угля.]''')
        st.divider()

        if choose_model == 'GradientBoost':
            gradient_boost_model(coal_dataset, 'coal')
            check_box_user_data = st.sidebar.checkbox('Ввести свои данные')
            if check_box_user_data:
                user_df = user_data('coal')
                if st.button('START'):
                    gradient_boost_model(coal_dataset, 'coal', user_data=user_df, check=True)
        elif choose_model == 'RandomForest':
            random_forest_model(coal_dataset, 'coal')
            check_box_user_data = st.sidebar.checkbox('Ввести свои данные')
            if check_box_user_data:
                user_df = user_data('coal')
                if st.button('START'):
                    random_forest_model(coal_dataset, 'coal', user_data=user_df, check=True)
        elif choose_model == 'AdaBoost':
            ada_boost_model(coal_dataset, 'coal')
            check_box_user_data = st.sidebar.checkbox('Ввести свои данные')
            if check_box_user_data:
                user_df = user_data('coal')
                if st.button('START'):
                    ada_boost_model(coal_dataset, 'coal', user_data=user_df, check=True)
        elif choose_model == 'BaggingRegressor':
            bagging_reg_model(coal_dataset, 'coal')
            check_box_user_data = st.sidebar.checkbox('Ввести свои данные')
            if check_box_user_data:
                user_df = user_data('coal')
                if st.button('START'):
                    bagging_reg_model(coal_dataset, 'coal', user_data=user_df, check=True)
        elif choose_model == 'DecisionTreeRegressor':
            decision_tree_model(coal_dataset, 'coal')
            check_box_user_data = st.sidebar.checkbox('Ввести свои данные')
            if check_box_user_data:
                user_df = user_data('coal')
                if st.button('START'):
                    decision_tree_model(coal_dataset, 'coal', user_data=user_df, check=True)

    elif choose_mineral == "Газ":
        choose_model = st.sidebar.selectbox("Выберите модель машинного обучения", [
            "No model", "AdaBoost", "RandomForest", "GradientBoost", "BaggingRegressor", "DecisionTreeRegressor"])

        st.title(':blue[Газовая промышленность]')
        st.divider()
        st.markdown(''':gray[Отрасль топливной промышленности, задача которой — добыча и разведка природного газа,
                 транспортировка по газопроводам, производство искусственного газа из угля и сланцев,
                  переработка природного газа, использование его в различных отраслях промышленности и
                   коммунально-бытовом хозяйстве.]''')
        st.divider()

        if choose_model == 'GradientBoost':
            gradient_boost_model(gas_dataset, 'gas')
            check_box_user_data = st.sidebar.checkbox('Ввести свои данные')
            if check_box_user_data:
                user_df = user_data('gas')
                if st.button('START'):
                    gradient_boost_model(gas_dataset, 'gas', user_data=user_df, check=True)

        elif choose_model == 'RandomForest':
            random_forest_model(gas_dataset, 'gas')
            check_box_user_data = st.sidebar.checkbox('Ввести свои данные')
            if check_box_user_data:
                user_df = user_data('gas')
                if st.button('START'):
                    random_forest_model(gas_dataset, 'gas', user_data=user_df, check=True)

        elif choose_model == 'AdaBoost':
            ada_boost_model(gas_dataset, 'gas')
            check_box_user_data = st.sidebar.checkbox('Ввести свои данные')
            if check_box_user_data:
                user_df = user_data('gas')
                if st.button('START'):
                    ada_boost_model(gas_dataset, 'gas', user_data=user_df, check=True)

        elif choose_model == 'BaggingRegressor':
            bagging_reg_model(gas_dataset, 'gas')
            check_box_user_data = st.sidebar.checkbox('Ввести свои данные')
            if check_box_user_data:
                user_df = user_data('gas')
                if st.button('START'):
                    bagging_reg_model(gas_dataset, 'gas', user_data=user_df, check=True)

        elif choose_model == 'DecisionTreeRegressor':
            decision_tree_model(gas_dataset, 'gas')
            check_box_user_data = st.sidebar.checkbox('Ввести свои данные')
            if check_box_user_data:
                user_df = user_data('gas')
                if st.button('START'):
                    decision_tree_model(gas_dataset, 'gas', user_data=user_df, check=True)


if __name__ == "__main__":
    main()
