from catboost.core import json
from feature_engine.imputation import MeanMedianImputer
from sklearn.impute import SimpleImputer
import pickle
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import label_binarize
from pathlib import Path
from tqdm import tqdm
tqdm.pandas()
import time

from . import modelGlobs

from datetime import datetime, timedelta
import calendar
import holidays

import warnings
from sklearn.exceptions import ConvergenceWarning

# Игнорирование предупреждений
warnings.simplefilter('ignore')
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Установка начального значения для генератора случайных чисел
seed = 42
np.random.seed(seed)


class ScoringModel:
    def __init__(self, weights_path: Path) -> None:
        """ Инициализация модели ScoringModel: загрузка предобученной модели и импера. """
        
        # Путь к сохраненной модели и имперу
        sub_model_rfc_path = weights_path / "RandomForestClassifier_scoring.pickle"
        sub_model_cat_path = weights_path / "CatBoostClassifier_scoring.pickle"
        sub_model_xgb_path = weights_path / "XGBClassifier_scoring.pickle"
        meanMedianImputer_path = weights_path / "MeanMedianImputer.pickle"
        graph_path = weights_path / "graph.csv"
        data_columns_path = weights_path / "data_columns.json"
        rfc_graph_scoring_path = weights_path / "RandomForestClassifier_SUB_model_scoring.pickle"
        unic_data_path = weights_path / "unic_data.csv"
        distance_dict_path = weights_path / "distance_dict.json"

        # Загрузка модели и импутера из файлов
        with open(sub_model_rfc_path, "rb") as fd:
            self.__sub_model_rfc = pickle.load(fd)
        with open(sub_model_cat_path, "rb") as fd:
            self.__sub_model_cat = pickle.load(fd)
        with open(sub_model_xgb_path, "rb") as fd:
            self.__sub_model_xgb = pickle.load(fd)
        with open(meanMedianImputer_path, "rb") as fd:
            self.__imputer = pickle.load(fd)
        with open(rfc_graph_scoring_path, "rb") as fd:
            self.__rfc_graph_scoring = pickle.load(fd)
        self.__graph = pd.read_csv(graph_path).drop(columns=['Unnamed: 0'])
        with open(data_columns_path, "r") as fd:
            self.__data_columns = list(json.load(fd).keys())
        self.__unic_data_df = pd.read_csv(unic_data_path)
        with open(distance_dict_path, "r") as fd:
            self.__distance_dict: dict = json.load(fd)
            self.__distance_dict = {int(k): v for k, v in self.__distance_dict.items()}


        self.__data_del_par = modelGlobs.inference_data_del_par
        self.__nan_par = modelGlobs.inference_nan_par
        self.__final_cols = modelGlobs.inference_final_cols
        self.__graph_del_col = modelGlobs.graph_del_col
        self.__graph_x_cols = modelGlobs.graph_x_cols

        # Инициализация российских праздников
        self.__ru_holidays = holidays.Russia()

    def preproc_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """ 
        Предобработка входных данных для предсказания:
        - Удаление ненужных колонок
        - Создание временных признаков
        - Импутация отсутствующих значений
        - Удаление дополнительных колонок
        - Создание расширенных признаков
        - Снижение использования памяти
        """
        
        return data
    

    def predict_base_model(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """ 
        Предсказание вероятностей на основе предобработанных данных
        с помощью блендинга базовых моделей.
        """

        data = test_df.copy()

        # Удаление колонок с пустыми или неинформативными данными
        data = data.drop(columns=self.__data_del_par)

        # Импутация отсутствующих значений
        data[self.__nan_par] = self.__imputer.transform(data[self.__nan_par])
        print("imp")
        

        # Создание временных признаков
        data = self.create_time_features(data, 'contract_date', 'report_date')
        print("time")

        # Удаление колонок, не нужных для предсказания модели
        data = data.drop(columns=['contract_date', 'contract_date', 'report_date'])

        # Создание расширенных признаков
        data = self.create_advanced_features(data)

        data = self.work_graph(data)

        # Снижение использования памяти
        data = self.reduce_mem_usage(data)

        data = pd.get_dummies(data, columns=['specialization_id', 'project_id', 'building_id', 'contractor_id'], dtype=int)
        null_cols = set(self.__data_columns) - set(data.columns)
        data[list(null_cols)] = 0

        # Предсказание вероятностей для положительного класса
        pred_rfc = self.__sub_model_rfc.predict_proba(data[self.__final_cols])[:, 1]
        pred_cat = self.__sub_model_cat.predict_proba(data[self.__final_cols])[:, 1]
        pred_xgb = self.__sub_model_xgb.predict_proba(data[self.__final_cols])[:, 1]

        # Комбинирование предсказаний
        pred = pred_rfc * 0.9 + pred_xgb * 0.05 + pred_cat * 0.05

        result = pd.DataFrame({'contract_id': test_df['contract_id'], 'report_date': test_df['report_date'], 'score': pred})
        result = result.merge(result.groupby('contract_id')['score'].max().reset_index(), how="left", on="contract_id")
        result = result[['contract_id', 'report_date', 'score_y']].rename(columns={'score_y': 'score'})

        return result

    def predict_graph_scoring(self, test_df: pd.DataFrame, preds_df: pd.DataFrame) -> pd.DataFrame:
        data = test_df.copy()
        data = data.drop(columns=['contract_id', 'project_id', 'building_id'])
        data[self.__nan_par] = self.__imputer.transform(data[self.__nan_par])
        data.fillna(0, inplace=True)
        data = data.drop(columns=self.__graph_del_col)
        data = self.reduce_mem_usage(data)
        data = data.join(preds_df, rsuffix="_r")
        
        unic_data = self.__unic_data_df
        distance_dict = self.__distance_dict
        unic_data_gb = unic_data.groupby(['contractor_id'], as_index=False).median()
        res = []

        to_cnt_i = list()
        to_rs2_i = list()

        data = data.set_index(['contractor_id', 'score'])[self.__graph_x_cols]

        for i, ((cn_id, scr), sf) in tqdm(enumerate(data.iterrows()), total=len(data)):
            if cn_id in distance_dict:
                d = unic_data_gb[unic_data.contractor_id == distance_dict[cn_id][0]]
                res2 = d.filter(like='contractor')
                if len(res2) != 0:
                    to_cnt_i.append(i)
                    to_rs2_i.append(res2.index[0])

            res.append(scr)
        
        res = np.array(res)
        tcn_df = data.iloc[to_cnt_i].copy().reset_index()
        tcn_df = tcn_df.filter(regex='^(?!.*contractor)', axis=1)
        tcn_rsd = unic_data_gb.loc[to_rs2_i]
        tcn_rsd = tcn_rsd.filter(like='contractor')      
        tcn_df[tcn_rsd.columns] = tcn_rsd.reset_index()[tcn_rsd.columns]
        preds_rfc = self.__rfc_graph_scoring.predict_proba(tcn_df[self.__graph_x_cols])[:, 1]
        preds_all = np.max([preds_rfc, res[to_cnt_i]], axis=0)
        res[to_cnt_i] = preds_all

        preds_df['score'] = res
        return preds_df

    def predict_scoring(self, data: pd.DataFrame) -> np.ndarray:
        preds_df = self.predict_base_model(data)
        preds_df = self.predict_graph_scoring(data, preds_df)

        return preds_df.score.values

    # Функция для определения сезона по месяцу даты
    # Принимает на вход объект даты (date) и возвращает строку с названием сезона
    def get_season(self, date) -> str:
        if date.month in [12, 1, 2]:
            return 'winter'  # Зима: декабрь, январь, февраль
        elif date.month in [3, 4, 5]:
            return 'spring'  # Весна: март, апрель, май
        elif date.month in [6, 7, 8]:
            return 'summer'  # Лето: июнь, июль, август
        else:
            return 'autumn'  # Осень: сентябрь, октябрь, ноябрь

    # Основная функция для добавления временных признаков в DataFrame
    # df - входной DataFrame, start_col - столбец с начальной датой, end_col - столбец с конечной датой
    def create_time_features(self, df: pd.DataFrame, start_col: str, end_col: str) -> pd.DataFrame:
        # Преобразуем столбцы с датами в формат datetime для работы с ними как с датами
        df[start_col] = pd.to_datetime(df[start_col])
        df[end_col] = pd.to_datetime(df[end_col])
        
        # Создаем новый столбец, который содержит диапазон всех дат от начальной до конечной
        df['date_range'] = df.apply(lambda row: pd.date_range(start=row[start_col], end=row[end_col]), axis=1)
        
        # Создаем столбец с общим количеством дней в диапазоне
        df['total_days'] = df['date_range'].apply(len)
        
        # Считаем долю зимних дней в диапазоне и нормализуем на общее количество дней
        df['winter'] = df['date_range'].apply(lambda dates: sum(self.get_season(date) == 'winter' for date in dates)) / df['total_days']
        
        # Считаем долю весенних дней в диапазоне и нормализуем на общее количество дней
        df['spring'] = df['date_range'].apply(lambda dates: sum(self.get_season(date) == 'spring' for date in dates)) / df['total_days']
        
        # Считаем долю летних дней в диапазоне и нормализуем на общее количество дней
        df['summer'] = df['date_range'].apply(lambda dates: sum(self.get_season(date) == 'summer' for date in dates)) / df['total_days']
        
        # Считаем долю осенних дней в диапазоне и нормализуем на общее количество дней
        df['autumn'] = df['date_range'].apply(lambda dates: sum(self.get_season(date) == 'autumn' for date in dates)) / df['total_days']
        
        # Считаем долю праздничных дней в диапазоне на основе списка российских праздников
        df['holidays'] = df['date_range'].apply(lambda dates: sum(date in self.__ru_holidays for date in dates)) / df['total_days']
        
        # Считаем долю выходных дней (суббота и воскресенье) и нормализуем на общее количество дней
        df['weekends'] = df['date_range'].apply(lambda dates: sum(date.weekday() >= 5 for date in dates)) / df['total_days']
        
        # Считаем долю рабочих дней (понедельник-пятница) с учетом того, что праздники также считаются нерабочими
        df['workdays'] = df['date_range'].apply(lambda dates: sum(date.weekday() < 5 for date in dates) - sum(date in self.__ru_holidays for date in dates)) / df['total_days']
        
        # Функция для подсчета количества "длинных" выходных
        # Длинные выходные определяются как выходные дни, соединенные с праздничными днями
        def count_long_weekends(dates):
            long_weekends = 0
            for i in range(1, len(dates)):  # Начинаем со второго дня диапазона
                # Если предыдущий день был выходным, а текущий — праздничным, увеличиваем счетчик длинных выходных
                if dates[i - 1].weekday() >= 5 and dates[i] in self.__ru_holidays:
                    long_weekends += 1
            return long_weekends
        
        # Добавляем столбец с долей длинных выходных дней в общем количестве дней
        df['long_weekends'] = df['date_range'].apply(lambda dates: count_long_weekends(dates)) / df['total_days']
        
        # Нормализуем общее количество дней на 366, чтобы учесть високосные годы
        df['total_days'] = df['total_days'] / 366

        return df  # Возвращаем DataFrame с новыми признаками 

    def reduce_mem_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Проходит по всем столбцам DataFrame и изменяет тип данных
            для уменьшения использования памяти.
        """
        start_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

        for col in df.columns:
            col_type = df[col].dtype.name

            if col_type not in ['object', 'category', 'datetime64[ns, UTC]']:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)

        end_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

        return df

    # Функция для создания дополнительных признаков на основе финансовых и других данных
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        # Финансовые коэффициенты
        
        # Текущая ликвидность (Current Ratio)
        # Показывает отношение текущих активов компании к её краткосрочным обязательствам.
        # Формула: Текущие активы / Краткосрочные обязательства
        df['current_ratio'] = df['agg_Finance__g_contractor__Value__CurrentAssets__last__ALL_TIME'] / df['agg_Finance__g_contractor__Value__ShortLiabilities__last__ALL_TIME']
        
        # Соотношение долга к капиталу (Debt to Equity Ratio)
        # Показывает соотношение долгов компании (краткосрочные и долгосрочные обязательства) к её собственному капиталу.
        # Формула: (Краткосрочные обязательства + Долгосрочные обязательства) / Собственный капитал
        df['debt_to_equity'] = (df['agg_Finance__g_contractor__Value__ShortLiabilities__last__ALL_TIME'] + df['agg_Finance__g_contractor__Value__LongLiabilities__last__ALL_TIME']) / df['agg_Finance__g_contractor__Value__Capital__last__ALL_TIME']
        
        # Рентабельность активов (Return on Assets, ROA)
        # Показывает, насколько эффективно компания использует свои активы для получения прибыли.
        # Формула: Чистая прибыль / Общие активы
        df['return_on_assets'] = df['agg_Finance__g_contractor__Value__NetProfit_y__last__ALL_TIME'] / df['agg_Finance__g_contractor__Value__Balance__last__ALL_TIME']
        
        # Динамика изменения финансовых и судебных показателей
        
        # Изменение сумм исков против компании за последние 12 месяцев по сравнению с предыдущими 12-24 месяцами
        df['claims_change_12_24'] = df['agg_ArbitrationCases__g_contractor__DefendantSum__sum__12M'] - df['agg_ArbitrationCases__g_contractor__DefendantSum__sum__12_24M']
        
        # Изменение сумм исков против компании за предыдущие 12-24 месяца по сравнению с 24-36 месяцами
        df['claims_change_24_36'] = df['agg_ArbitrationCases__g_contractor__DefendantSum__sum__12_24M'] - df['agg_ArbitrationCases__g_contractor__DefendantSum__sum__12_36M']
        
        # Изменение прибыли компании относительно её выручки
        # Показывает, насколько прибыльной была компания относительно общего дохода (выручки).
        # Формула: Чистая прибыль / Выручка
        df['profit_change'] = df['agg_Finance__g_contractor__Value__NetProfit_y__last__ALL_TIME'] / df['agg_Finance__g_contractor__Value__Revenue_y__last__ALL_TIME']
        
        # Суммарные показатели и рейтинги
        
        # Индекс надежности компании
        # Рассчитывается как среднее значение нескольких индексов, которые отражают общее состояние компании,
        # её риск несостоятельности, платежеспособность и уровень добросовестности.
        df['reliability_index'] = (
            df['agg_ConsolidatedIndicator__g_contractor__Index__Overall__mean__ALL_TIME'] +  # Общий индекс компании
            df['agg_ConsolidatedIndicator__g_contractor__Index__FailureScore__mean__ALL_TIME'] +  # Индекс вероятности банкротства
            df['agg_ConsolidatedIndicator__g_contractor__Index__PaymentIndex__mean__ALL_TIME'] +  # Платежный индекс
            df['agg_ConsolidatedIndicator__g_contractor__Index__IndexOfDueDiligence__mean__ALL_TIME']  # Индекс добросовестности
        ) / 4  # Среднее значение этих индексов
        
        # Активность по залогам
        
        # Активность по залогам — разница между количеством активных залогов и прекращённых залогов.
        df['pledger_activity'] = df['agg_spark_extended_report__g_contractor__PledgerActiveCount__last__ALL_TIME'] - df['agg_spark_extended_report__g_contractor__PledgerCeasedCount__last__ALL_TIME']
        
        # Исторические данные по искам
        
        # Общая сумма исков за последние 12 месяцев (ответчик и истец)
        df['total_claims_last_12_months'] = df['agg_ArbitrationCases__g_contractor__DefendantSum__sum__12M'] + df['agg_ArbitrationCases__g_contractor__PlaintiffSum__sum__12M']
        
        # Общая сумма исков за последние 24 месяца (ответчик и истец)
        df['total_claims_last_24_months'] = df['agg_ArbitrationCases__g_contractor__DefendantSum__sum__12_24M'] + df['agg_ArbitrationCases__g_contractor__PlaintiffSum__sum__12_24M']
        
        # Логарифмическое преобразование
        
        # Логарифм от оценочной суммы обязательств компании
        df['agg_spark_extended_report__g_contractor__EstimatedLiabilitiesSum__last__ALL_TIME'] = np.log1p(df['agg_spark_extended_report__g_contractor__EstimatedLiabilitiesSum__last__ALL_TIME'])
        
        # Логарифм от выручки компании за последний год
        df['agg_Finance__g_contractor__Value__Revenue_y__last__ALL_TIME'] = np.log1p(df['agg_Finance__g_contractor__Value__Revenue_y__last__ALL_TIME'])
        
        # Новые финансовые признаки
        
        # Абсолютное изменение суммы контракта (текущая сумма минус начальная сумма)
        df['contract_sum_change'] = df['contract_current_sum'] - df['contract_init_sum']
        
        # Процентное изменение суммы контракта (относительное изменение текущей суммы к начальной)
        # Если значение бесконечно (например, начальная сумма контракта была 0), заменяем его на NaN
        df['contract_sum_change_percentage'] = (df['contract_sum_change'] / df['contract_init_sum']).replace([np.inf, -np.inf], np.nan)
        
        # Логарифм от текущей суммы контракта (логарифмическое преобразование для нормализации данных)
        df['contract_current_sum'] = np.log1p(df['contract_current_sum'])
        
        # Нормализуем изменение суммы контракта по общему количеству контрактов
        # Делим изменение суммы контракта на количество контрактов, чтобы получить среднее изменение на контракт
        df['contract_change_per_contract'] = df['contract_sum_change'] / df['agg_all_contracts__g_contract__bit_da_guid__isMain__count__ALL_TIME']
        
        return df  # Возвращаем DataFrame с новыми признаками 


    def work_graph(self, df: pd.DataFrame) -> pd.DataFrame:
        # Создаем словари для contractor_id1
        contractor1_group = self.__graph.groupby('contractor_id1')['Distance']
        
        contractor1_min = contractor1_group.min().to_dict()
        contractor1_max = contractor1_group.max().to_dict()
        contractor1_mean = contractor1_group.mean().to_dict()
        contractor1_median = contractor1_group.median().to_dict()
        contractor1_sum = contractor1_group.sum().to_dict()
        contractor1_count = contractor1_group.count().to_dict()

        # Создаем словари для contractor_id2
        contractor2_group = self.__graph.groupby('contractor_id2')['Distance']
        
        contractor2_min = contractor2_group.min().to_dict()
        contractor2_max = contractor2_group.max().to_dict()
        contractor2_mean = contractor2_group.mean().to_dict()
        contractor2_median = contractor2_group.median().to_dict()
        contractor2_sum = contractor2_group.sum().to_dict()
        contractor2_count = contractor2_group.count().to_dict()

        # Используем словари для ускорения поиска
        df['Distance_to_contractor_min'] = df['contractor_id'].map(contractor1_min).fillna(-1)
        df['Distance_to_contractor_max'] = df['contractor_id'].map(contractor1_max).fillna(-1)
        df['Distance_to_contractor_mean'] = df['contractor_id'].map(contractor1_mean).fillna(-1)
        df['Distance_to_contractor_median'] = df['contractor_id'].map(contractor1_median).fillna(-1)
        df['Distance_to_contractor_sum'] = df['contractor_id'].map(contractor1_sum).fillna(-1)
        df['Distance_to_contractor_count'] = df['contractor_id'].map(contractor1_count).fillna(0)
        
        df['Distance_from_contractor_min'] = df['contractor_id'].map(contractor2_min).fillna(-1)
        df['Distance_from_contractor_max'] = df['contractor_id'].map(contractor2_max).fillna(-1)
        df['Distance_from_contractor_mean'] = df['contractor_id'].map(contractor2_mean).fillna(-1)
        df['Distance_from_contractor_median'] = df['contractor_id'].map(contractor2_median).fillna(-1)
        df['Distance_from_contractor_sum'] = df['contractor_id'].map(contractor2_sum).fillna(-1)
        df['Distance_from_contractor_count'] = df['contractor_id'].map(contractor2_count).fillna(0)

        return df
	
# Used for testing purpose
def _test():
    test_df = pd.read_csv(Path(__file__).parent / "../test2_X.csv")
    subm_df = pd.read_csv(Path(__file__).parent / "../subm.csv").drop(columns=["Unnamed: 0"])
    model = ScoringModel(Path(__file__).parent / "../weights")
    preds = model.predict_graph_scoring(test_df, subm_df)
    subm = preds
    # subm = pd.DataFrame({"contract_id": test_df['contract_id'], 'report_date': test_df['report_date'], 'score': preds.score})
    print(subm)
    subm.to_csv(Path(__file__).parent / "../submm.csv", index=False)
    
if __name__ == "__main__":
    _test()
