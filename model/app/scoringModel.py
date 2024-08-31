from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import holidays
import pickle
import warnings
from sklearn.exceptions import ConvergenceWarning

from .config import get_settings

warnings.simplefilter('ignore')
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
seed = 42
np.random.seed(seed)

class ScoringModel:
    def __init__(self) -> None:
        """
        Initialize the ScoringModel by loading pre-trained model and imputer.
        """
        # Paths to saved model and imputer
        model_path = get_settings().scoring_model_path
        meanMedianImputer_path = get_settings().scoring_mean_median_imputer_path

        # Load model and imputer from pickle files
        self.__model = pickle.load(open(model_path, "rb"))
        self.__imputer = pickle.load(open(meanMedianImputer_path, "rb"))
        
        # Initialize Russian holidays
        self.__ru_holidays = holidays.Russia() # pyright: ignore[reportAttributeAccessIssue]

        # Columns to drop as they are empty or not useful
        self.__empty_par = [
            'agg_all_contracts__g_contract__abs_change_price_last_ds__isMain__last__ALL_TIME',
            'agg_all_contracts__g_contract__abs_change_price_last_ds__isMain__mean__ALL_TIME',
            'agg_all_contracts__g_contract__rel_change_price_last_ds__isMain__last__ALL_TIME',
            'agg_all_contracts__g_contract__rel_change_price_last_ds__isMain__mean__ALL_TIME',
            'agg_FinanceAndTaxesFTS__g_contractor__TaxPenaltiesSum__last__ALL_TIME'
        ]

        # Columns to drop for data preprocessing
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

    def __preproc_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess the input data for prediction:
        - Drop unnecessary columns
        - Create time-related features
        - Impute missing values
        - Drop additional columns
        - Create advanced features
        - Reduce memory usage
        """
        # Drop columns with empty or non-informative data
        data = data.drop(columns=self.__empty_par)
        
        # Create time-related features from date columns
        data = self.__create_time_features(data, 'contract_date', 'report_date')
        
        # Drop columns not needed for model prediction
        data = data.drop(columns=['report_date', 'contract_date', 'date_range'])
        
        # Impute missing values
        data = self.__imputer.transform(data)
        
        # Drop additional columns specific to data preprocessing
        data = data.drop(columns=self.__data_del_par)
        
        # Create advanced features from remaining data
        data = self.__create_advanced_features(data)
        
        # Normalize the specialization_id feature
        data['specialization_id'] = data['specialization_id'] / 100
        
        # Reduce memory usage of the DataFrame
        data = self.__reduce_mem_usage(data)
        
        # Convert DataFrame to NumPy array for model input
        return data.to_numpy()

    def predict_scoring(self, predata: pd.DataFrame) -> np.ndarray:
        """
        Predict the scoring using the preprocessed data:
        - Process the input data
        - Predict probabilities with the loaded model
        """
        # Preprocess the data
        data = self.__preproc_data(predata)
        
        # Predict probabilities for the positive class
        pred = self.__model.predict_proba(data)[:, 1]
        
        return pred

    @staticmethod
    def __reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce memory usage by optimizing data types of DataFrame columns.
        """
        # Calculate and print initial memory usage
        start_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

        # Iterate through columns and change data types
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

        # Calculate and print reduced memory usage
        end_mem = df.memory_usage().sum() / 1024 ** 2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

        return df

    @staticmethod
    def __get_season(date):
        """
        Determine the season for a given date.
        """
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
        Create time-related features from start and end date columns.
        """
        # Convert date columns to datetime format
        df[start_col] = pd.to_datetime(df[start_col])
        df[end_col] = pd.to_datetime(df[end_col])
        
        # Calculate the date range between start and end dates
        df['date_range'] = df.apply(lambda row: pd.date_range(start=row[start_col], end=row[end_col]), axis=1)
        df['total_days'] = df['date_range'].apply(len)  # Total number of days in the range
        
        # Calculate normalized seasonal distributions
        df['winter'] = df['date_range'].apply(lambda dates: sum(self.__get_season(date) == 'winter' for date in dates)) / df['total_days']
        df['spring'] = df['date_range'].apply(lambda dates: sum(self.__get_season(date) == 'spring' for date in dates)) / df['total_days']
        df['summer'] = df['date_range'].apply(lambda dates: sum(self.__get_season(date) == 'summer' for date in dates)) / df['total_days']
        df['autumn'] = df['date_range'].apply(lambda dates: sum(self.__get_season(date) == 'autumn' for date in dates)) / df['total_days']
        
        # Calculate normalized counts of holidays and weekends
        df['holidays'] = df['date_range'].apply(lambda dates: sum(date in self.__ru_holidays for date in dates)) / df['total_days']
        df['weekends'] = df['date_range'].apply(lambda dates: sum(date.weekday() >= 5 for date in dates)) / df['total_days']
        df['workdays'] = df['date_range'].apply(lambda dates: sum(date.weekday() < 5 for date in dates) - sum(date in self.__ru_holidays for date in dates)) / df['total_days']
        
        # Count long weekends (weekends that are connected to holidays)
        def count_long_weekends(dates):
            long_weekends = 0
            for i in range(1, len(dates)):
                if dates[i - 1].weekday() >= 5 and dates[i] in self.__ru_holidays:
                    long_weekends += 1
            return long_weekends
        
        df['long_weekends'] = df['date_range'].apply(lambda dates: count_long_weekends(dates)) / df['total_days']
        
        # Normalize total days to a year
        df['total_days'] = df['total_days'] / 366
        
        return df    

    @staticmethod
    def __create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced features based on financial and historical data.
        """
        # Financial ratios
        df['current_ratio'] = df['agg_Finance__g_contractor__Value__CurrentAssets__last__ALL_TIME'] / df['agg_Finance__g_contractor__Value__ShortLiabilities__last__ALL_TIME']
        df['debt_to_equity'] = (df['agg_Finance__g_contractor__Value__ShortLiabilities__last__ALL_TIME'] + df['agg_Finance__g_contractor__Value__LongLiabilities__last__ALL_TIME']) / df['agg_Finance__g_contractor__Value__Capital__last__ALL_TIME']
        df['return_on_assets'] = df['agg_Finance__g_contractor__Value__NetProfit_y__last__ALL_TIME'] / df['agg_Finance__g_contractor__Value__Balance__last__ALL_TIME']
        
        # Changes in claims and financial indicators
        df['claims_change_12_24'] = df['agg_ArbitrationCases__g_contractor__DefendantSum__sum__12M'] - df['agg_ArbitrationCases__g_contractor__DefendantSum__sum__12_24M']
        df['claims_change_24_36'] = df['agg_ArbitrationCases__g_contractor__DefendantSum__sum__12_24M'] - df['agg_ArbitrationCases__g_contractor__DefendantSum__sum__12_36M']
        df['profit_change'] = df['agg_Finance__g_contractor__Value__NetProfit_y__last__ALL_TIME'] / df['agg_Finance__g_contractor__Value__Revenue_y__last__ALL_TIME']
        
        # Aggregated indicators and ratings
        df['reliability_index'] = (
            df['agg_ConsolidatedIndicator__g_contractor__Index__Overall__mean__ALL_TIME'] +
            df['agg_ConsolidatedIndicator__g_contractor__Index__FailureScore__mean__ALL_TIME'] +
            df['agg_ConsolidatedIndicator__g_contractor__Index__PaymentIndex__mean__ALL_TIME'] +
            df['agg_ConsolidatedIndicator__g_contractor__Index__IndexOfDueDiligence__mean__ALL_TIME']
        ) / 4
        
        # Pledger activity
        df['pledger_activity'] = df['agg_spark_extended_report__g_contractor__PledgerActiveCount__last__ALL_TIME'] - df['agg_spark_extended_report__g_contractor__PledgerCeasedCount__last__ALL_TIME']
        
        # Historical claims data
        df['total_claims_last_12_months'] = df['agg_ArbitrationCases__g_contractor__DefendantSum__sum__12M'] + df['agg_ArbitrationCases__g_contractor__PlaintiffSum__sum__12M']
        df['total_claims_last_24_months'] = df['agg_ArbitrationCases__g_contractor__DefendantSum__sum__12_24M'] + df['agg_ArbitrationCases__g_contractor__PlaintiffSum__sum__12_24M']
        
        # Logarithmic transformations for skewed data
        df['agg_spark_extended_report__g_contractor__EstimatedLiabilitiesSum__last__ALL_TIME'] = np.log1p(df['agg_spark_extended_report__g_contractor__EstimatedLiabilitiesSum__last__ALL_TIME'])
        df['agg_Finance__g_contractor__Value__Revenue_y__last__ALL_TIME'] = np.log1p(df['agg_Finance__g_contractor__Value__Revenue_y__last__ALL_TIME'])
        
        # Contract sum changes
        df['contract_sum_change'] = df['contract_current_sum'] - df['contract_init_sum']
        df['contract_sum_change_percentage'] = (df['contract_sum_change'] / df['contract_init_sum']).replace([np.inf, -np.inf], np.nan)
        
        # Normalize contract price changes
        df['contract_change_per_contract'] = df['contract_sum_change'] / df['agg_all_contracts__g_contract__bit_da_guid__isMain__count__ALL_TIME']
        
        # Binary indicators for financial conditions
        df['Income > Expenses'] = df['agg_FinanceAndTaxesFTS__g_contractor__Income__last__ALL_TIME'] > df['agg_FinanceAndTaxesFTS__g_contractor__Expenses__last__ALL_TIME']
        df['Income > Expenses'] = df['Income > Expenses'].map(int)
        df['Net Profit Positive'] = df['agg_Finance__g_contractor__Value__NetProfit_y__last__ALL_TIME'] > 0
        df['Net Profit Positive'] = df['Net Profit Positive'].map(int)
        df['Current Assets > Long Liabilities'] = df['agg_Finance__g_contractor__Value__CurrentAssets__last__ALL_TIME'] > df['agg_Finance__g_contractor__Value__LongLiabilities__last__ALL_TIME']
        df['Current Assets > Long Liabilities'] = df['Current Assets > Long Liabilities'].map(int)
        df['Credit Limit > Expenses'] = df['agg_spark_extended_report__g_contractor__CreditLimitSum__last__ALL_TIME'] > df['agg_FinanceAndTaxesFTS__g_contractor__Expenses__last__ALL_TIME']
        df['Credit Limit > Expenses'] = df['Credit Limit > Expenses'].map(int)
        df['Revenue > Taxes'] = df['agg_spark_extended_report__g_contractor__CompanySizeRevenue__last__ALL_TIME'] > df['agg_FinanceAndTaxesFTS__g_contractor__TaxesSum__last__ALL_TIME']
        df['Revenue > Taxes'] = df['Revenue > Taxes'].map(int)
        df['Income > Taxes + Expenses'] = df['agg_FinanceAndTaxesFTS__g_contractor__Income__last__ALL_TIME'] > (df['agg_FinanceAndTaxesFTS__g_contractor__TaxesSum__last__ALL_TIME'] + df['agg_FinanceAndTaxesFTS__g_contractor__Expenses__last__ALL_TIME'])
        df['Income > Taxes + Expenses'] = df['Income > Taxes + Expenses'].map(int)

        # Aggregate financial indicators into a single score
        df['point_Finance'] = (df['Income > Taxes + Expenses'] + df['Revenue > Taxes'] +
                                    df['Credit Limit > Expenses'] + df['Current Assets > Long Liabilities'] +
                                    df['Net Profit Positive'] + df['Income > Expenses']) / 6
        
        return df
