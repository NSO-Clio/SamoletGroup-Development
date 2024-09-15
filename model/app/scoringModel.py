
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import holidays
import pickle
import warnings
from sklearn.exceptions import ConvergenceWarning

from .config import get_settings

# Игнорирование предупреждений
warnings.simplefilter('ignore')
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Установка начального значения для генератора случайных чисел
seed = 42
np.random.seed(seed)


class ScoringModel:
    def __init__(self) -> None:
        """ Инициализация модели ScoringModel: загрузка предобученной модели и импера. """
        # Путь к сохраненной модели и имперу
        model_path = get_settings().scoring_model_path
        sub_model_cat_path = get_settings().scoring_cbc_path
        sub_model_xgb_path = get_settings().scoring_xgbc_path
        meanMedianImputer_path = get_settings().scoring_mean_median_imputer_path

        # Загрузка модели и импера из файлов
        self.__model = pickle.load(open(model_path, "rb"))
        self.__sub_model_cat = pickle.load(open(sub_model_cat_path, "rb"))
        self.__sub_model_xgb = pickle.load(open(sub_model_xgb_path, "rb"))
        self.__imputer = pickle.load(open(meanMedianImputer_path, "rb"))

        # Инициализация российских праздников
        self.__ru_holidays = holidays.Russia()

        # Колонки, которые нужно удалить
        self.__empty_par = [
            'agg_all_contracts__g_contract__abs_change_price_last_ds__isMain__last__ALL_TIME',
            'agg_all_contracts__g_contract__abs_change_price_last_ds__isMain__mean__ALL_TIME',
            'agg_all_contracts__g_contract__rel_change_price_last_ds__isMain__last__ALL_TIME',
            'agg_all_contracts__g_contract__rel_change_price_last_ds__isMain__mean__ALL_TIME',
            'agg_FinanceAndTaxesFTS__g_contractor__TaxPenaltiesSum__last__ALL_TIME'
        ]

        self.__nan_par = [
            'agg_scontrol__g_contractor__close_delay__defect_type_repair__mean__ALL_TIME',
            'agg_scontrol__g_contractor__close_delay__defect_type_author_supervision__mean__ALL_TIME',
            'agg_scontrol__g_contractor__close_delay__defect_type_GR__mean__ALL_TIME',
            'agg_scontrol__g_contractor__close_delay__defect_type_labour_protection__mean__ALL_TIME',
            'agg_scontrol__g_contractor__close_delay__defect_type_general_contractor__mean__ALL_TIME',
            'agg_scontrol__g_contractor__close_delay__defect_type_arch_supervision__mean__ALL_TIME',
            'agg_scontrol__g_contractor__close_delay__defect_type_tech_supervision__mean__ALL_TIME',
            'agg_scontrol__g_contractor__close_delay__defect_type_app__mean__ALL_TIME',
            'agg_sroomer__g_contractor__sroomer_id__count__3M',
            'agg_sroomer__g_contractor__sroomer_id__count__6M',
            'agg_sroomer__g_contractor__sroomer_id__count__12M',
            'agg_sroomer__g_contractor__sroomer_id__count__ALL_TIME',
            'agg_BoardOfDirectors__g_contractor__Name__count__ALL_TIME',
            'agg_ConsolidatedIndicator__g_contractor__Index__Overall__mean__ALL_TIME',
            'agg_ConsolidatedIndicator__g_contractor__Index__FailureScore__mean__ALL_TIME',
            'agg_ConsolidatedIndicator__g_contractor__Index__PaymentIndex__mean__ALL_TIME',
            'agg_ConsolidatedIndicator__g_contractor__Index__IndexOfDueDiligence__mean__ALL_TIME',
            'agg_spark_extended_report__g_contractor__EstimatedClaimsSum__last__ALL_TIME',
            'agg_spark_extended_report__g_contractor__EstimatedLiabilitiesSum__last__ALL_TIME',
            'agg_spark_extended_report__g_contractor__EstimatedNetLiabilitiesSum__last__ALL_TIME',
            'agg_spark_extended_report__g_contractor__PledgeeActiveCount__last__ALL_TIME',
            'agg_spark_extended_report__g_contractor__PledgeeCeasedCount__last__ALL_TIME',
            'agg_spark_extended_report__g_contractor__PledgerActiveCount__last__ALL_TIME',
            'agg_spark_extended_report__g_contractor__PledgerCeasedCount__last__ALL_TIME',
            'agg_spark_extended_report__g_contractor__CompanySizeRevenue__last__ALL_TIME',
            'agg_spark_extended_report__g_contractor__CreditLimitSum__last__ALL_TIME',
            'agg_Finance__g_contractor__Value__FixedAssets__last__ALL_TIME',
            'agg_Finance__g_contractor__Value__CurrentAssets__last__ALL_TIME',
            'agg_Finance__g_contractor__Value__Capital__last__ALL_TIME',
            'agg_Finance__g_contractor__Value__LongLiabilities__last__ALL_TIME',
            'agg_Finance__g_contractor__Value__ShortLiabilities__last__ALL_TIME',
            'agg_Finance__g_contractor__Value__Balance__last__ALL_TIME',
            'agg_Finance__g_contractor__Value__Revenue_y__last__ALL_TIME',
            'agg_Finance__g_contractor__Value__NetProfit_y__last__ALL_TIME',
            'agg_Finance__g_contractor__Value__CostPrice_y__last__ALL_TIME',
            'agg_FinanceAndTaxesFTS__g_contractor__Expenses__last__ALL_TIME',
            'agg_FinanceAndTaxesFTS__g_contractor__Income__last__ALL_TIME',
            'agg_FinanceAndTaxesFTS__g_contractor__TaxArrearsSum__last__ALL_TIME',
            'agg_FinanceAndTaxesFTS__g_contractor__TaxesSum__last__ALL_TIME',
            'agg_ArbitrationCases__g_contractor__DefendantSum__sum__12M',
            'agg_ArbitrationCases__g_contractor__DefendantSum__sum__12_24M',
            'agg_ArbitrationCases__g_contractor__DefendantSum__sum__12_36M',
            'agg_ArbitrationCases__g_contractor__DefendantSum__sum__12_48M',
            'agg_ArbitrationCases__g_contractor__DefendantSum__sum__ALL_TIME',
            'agg_ArbitrationCases__g_contractor__PlaintiffSum__sum__12M',
            'agg_ArbitrationCases__g_contractor__PlaintiffSum__sum__12_24M',
            'agg_ArbitrationCases__g_contractor__PlaintiffSum__sum__12_36M',
            'agg_ArbitrationCases__g_contractor__PlaintiffSum__sum__12_48M',
            'agg_ArbitrationCases__g_contractor__PlaintiffSum__sum__ALL_TIME',
            'agg_tender_proposal__g_contractor__id__ALL__countDistinct__1W',
            'agg_tender_proposal__g_contractor__id__ALL__countDistinct__2W',
            'agg_tender_proposal__g_contractor__id__ALL__countDistinct__4W',
            'agg_tender_proposal__g_contractor__id__ALL__countDistinct__8W',
            'agg_tender_proposal__g_contractor__id__ALL__countDistinct__12W',
            'agg_tender_proposal__g_contractor__id__ALL__countDistinct__26W',
            'agg_tender_proposal__g_contractor__id__ALL__countDistinct__52W',
            'agg_tender_proposal__g_contractor__id__ALL__countDistinct__ALL_TIME'
        ]

        # Колонки для удаления в процессе обработки данных
        self.__data_del_par = [
            'project_id', 'contractor_id', 'building_id', 'contract_id',
            'agg_BoardOfDirectors__g_contractor__Name__count__ALL_TIME',
            'agg_cec_requests__g_contract__time_btw_requests__all__mean__1M',
            'agg_cec_requests__g_contract__time_btw_requests__all__mean__2M',
            'agg_cec_requests__g_contract__time_btw_requests__all__mean__3M',
            'agg_cec_requests__g_contract__time_btw_requests__all__mean__4M',
            'agg_cec_requests__g_contract__time_btw_requests__all__mean__5M',
            'agg_cec_requests__g_contract__time_btw_requests__all__mean__6M',
            'agg_cec_requests__g_contract__time_btw_requests__all__mean__7M',
            'agg_cec_requests__g_contract__time_btw_requests__all__mean__8M',
            'agg_cec_requests__g_contract__time_btw_requests__all__mean__12M',
            'agg_cec_requests__g_contract__time_btw_requests__all__mean__ALL_TIME',
            'agg_cec_requests__g_contract__created_dt__all__min__ALL_TIME',
            'agg_cec_requests__g_contract__created_dt__accepted__min__ALL_TIME',
            'agg_tender_proposal__g_contractor__id__ALL__countDistinct__1W',
            'agg_tender_proposal__g_contractor__id__ALL__countDistinct__2W',
            'agg_tender_proposal__g_contractor__id__ALL__countDistinct__4W',
            'agg_tender_proposal__g_contractor__id__ALL__countDistinct__8W',
            'agg_tender_proposal__g_contractor__id__ALL__countDistinct__26W',
            'agg_tender_proposal__g_contractor__id__ALL__countDistinct__52W',
            'agg_tender_proposal__g_contractor__id__ALL__countDistinct__ALL_TIME'
        ]

    def __preproc_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """ 
        Предобработка входных данных для предсказания:
        - Удаление ненужных колонок
        - Создание временных признаков
        - Импутация отсутствующих значений
        - Удаление дополнительных колонок
        - Создание расширенных признаков
        - Снижение использования памяти
        """
        # Удаление колонок с пустыми или неинформативными данными
        data = data.drop(columns=self.__empty_par)

        # Импутация отсутствующих значений
        data[self.__nan_par] = self.__imputer.transform(data[self.__nan_par])

        # Создание временных признаков
        data = self.__create_time_features(data, 'contract_date', 'report_date')

        # Удаление колонок, не нужных для предсказания модели
        data = data.drop(columns=['report_date', 'contract_date', 'date_range'])

        # Удаление дополнительных колонок
        data = data.drop(columns=self.__data_del_par)

        # Создание расширенных признаков
        data = self.__create_advanced_features(data)

        # Нормализация признака specialization_id
        data['specialization_id'] = data['specialization_id'] / 100

        # Снижение использования памяти
        data = self.__reduce_mem_usage(data)

        return data

    def predict_scoring(self, data: pd.DataFrame) -> np.ndarray:
        """ 
        Предсказание оценки на основе предобработанных данных:
        - Обработка входных данных
        - Предсказание вероятностей с загруженной моделью 
        """
        # Предобработка данных
        data = self.__preproc_data(data)

        # Предсказание вероятностей для положительного класса
        pred_model = self.__model.predict_proba(data)[:, 1]
        pred_cat = self.__sub_model_cat.predict_proba(data)[:, 1]
        pred_xgb = self.__sub_model_xgb.predict_proba(data)[:, 1]

        # Комбинирование предсказаний
        pred = pred_model * 0.9 + pred_xgb * 0.05 + pred_cat * 0.05

        return pred

    @staticmethod
    def __reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
        """ 
        Снижение использования памяти путем оптимизации типов данных колонок DataFrame.
        """
        # Расчет и вывод начального использования памяти
        start_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

        # Итерация по колонкам и изменение типов данных
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

        # Расчет и вывод уменьшенного использования памяти
        end_mem = df.memory_usage().sum() / 1024 ** 2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

        return df

    @staticmethod
    def __get_season(date: datetime) -> str:
        """ Определение сезона для данной даты. """
        if date.month in [12, 1, 2]:
            return 'winter'
        elif date.month in [3, 4, 5]:
            return 'spring'
        elif date.month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'

    def __create_time_features(self, df: pd.DataFrame, start_col: str, end_col: str) -> pd.DataFrame:
        """ 
        Создание временных признаков из колонок с начальной и конечной датами.
        """
        # Преобразование колонок дат в формат datetime
        df[start_col] = pd.to_datetime(df[start_col])
        df[end_col] = pd.to_datetime(df[end_col])

        # Расчет диапазона дат между начальной и конечной датами
        df['date_range'] = df.apply(lambda row: pd.date_range(start=row[start_col], end=row[end_col]), axis=1)
        df['total_days'] = df['date_range'].apply(len)

        # Расчет нормированных сезонных распределений
        df['winter'] = df['date_range'].apply(lambda dates: sum(self.__get_season(date) == 'winter' for date in dates)) / df['total_days']
        df['spring'] = df['date_range'].apply(lambda dates: sum(self.__get_season(date) == 'spring' for date in dates)) / df['total_days']
        df['summer'] = df['date_range'].apply(lambda dates: sum(self.__get_season(date) == 'summer' for date in dates)) / df['total_days']
        df['autumn'] = df['date_range'].apply(lambda dates: sum(self.__get_season(date) == 'autumn' for date in dates)) / df['total_days']

        # Расчет нормированных количеств праздников и выходных
        df['holidays'] = df['date_range'].apply(lambda dates: sum(date in self.__ru_holidays for date in dates)) / df['total_days']
        df['weekends'] = df['date_range'].apply(lambda dates: sum(date.weekday() >= 5 for date in dates)) / df['total_days']
        df['workdays'] = df['date_range'].apply(lambda dates: sum(date.weekday() < 5 for date in dates) - sum(date in self.__ru_holidays for date in dates)) / df['total_days']

        # Подсчет длинных выходных
        def count_long_weekends(dates):
            long_weekends = 0
            for i in range(1, len(dates)):
                if dates[i - 1].weekday() >= 5 and dates[i] in self.__ru_holidays:
                    long_weekends += 1
            return long_weekends

        df['long_weekends'] = df['date_range'].apply(lambda dates: count_long_weekends(dates)) / df['total_days']

        # Нормализация общего количества дней в году
        df['total_days'] = df['total_days'] / 366

        return df    

    @staticmethod
    def __create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:        
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
        
        return df
