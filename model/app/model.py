DATA_ALLOWED_COLUMNS = ['contract_id', 'report_date', 'specialization_id', 'contract_init_sum', 'contract_date', 'project_id', 'building_id', 'contractor_id', 'contract_current_sum', 'agg_all_contracts__g_contract__bit_da_guid__isMain__count__ALL_TIME', 'agg_all_contracts__g_contract__abs_change_price_last_ds__isMain__last__ALL_TIME', 'agg_all_contracts__g_contract__abs_change_price_last_ds__isMain__mean__ALL_TIME', 'agg_all_contracts__g_contract__rel_change_price_last_ds__isMain__last__ALL_TIME', 'agg_all_contracts__g_contract__rel_change_price_last_ds__isMain__mean__ALL_TIME', 'agg_cec_requests__g_contract__request_id__all__count__1W', 'agg_cec_requests__g_contract__request_id__all__count__2W', 'agg_cec_requests__g_contract__request_id__all__count__3W', 'agg_cec_requests__g_contract__request_id__all__count__4W', 'agg_cec_requests__g_contract__request_id__all__count__5W', 'agg_cec_requests__g_contract__request_id__all__count__6W', 'agg_cec_requests__g_contract__request_id__all__count__7W', 'agg_cec_requests__g_contract__request_id__all__count__8W', 'agg_cec_requests__g_contract__request_id__all__count__12W', 'agg_cec_requests__g_contract__request_id__all__count__ALL_TIME', 'counteragent_sum_agg_cec_requests__g_contract__request_id__all__count__ALL_TIME', 'agg_cec_requests__g_contract__total_sum_accepted__all__sum__1W', 'agg_cec_requests__g_contract__total_sum_accepted__all__sum__2W', 'agg_cec_requests__g_contract__total_sum_accepted__all__sum__3W', 'agg_cec_requests__g_contract__total_sum_accepted__all__sum__4W', 'agg_cec_requests__g_contract__total_sum_accepted__all__sum__5W', 'agg_cec_requests__g_contract__total_sum_accepted__all__sum__6W', 'agg_cec_requests__g_contract__total_sum_accepted__all__sum__7W', 'agg_cec_requests__g_contract__total_sum_accepted__all__sum__8W', 'agg_cec_requests__g_contract__total_sum_accepted__all__sum__12W', 'agg_cec_requests__g_contract__total_sum_accepted__all__sum__ALL_TIME', 'counteragent_sum_agg_cec_requests__g_contract__total_sum_accepted__all__sum__ALL_TIME', 'agg_cec_requests__g_contract__time_btw_requests__all__mean__1M', 'agg_cec_requests__g_contract__time_btw_requests__all__mean__2M', 'agg_cec_requests__g_contract__time_btw_requests__all__mean__3M', 'agg_cec_requests__g_contract__time_btw_requests__all__mean__4M', 'agg_cec_requests__g_contract__time_btw_requests__all__mean__5M', 'agg_cec_requests__g_contract__time_btw_requests__all__mean__6M', 'agg_cec_requests__g_contract__time_btw_requests__all__mean__7M', 'agg_cec_requests__g_contract__time_btw_requests__all__mean__8M', 'agg_cec_requests__g_contract__time_btw_requests__all__mean__12M', 'agg_cec_requests__g_contract__time_btw_requests__all__mean__ALL_TIME', 'agg_cec_requests__g_contract__created_dt__all__min__ALL_TIME', 'agg_cec_requests__g_contract__created_dt__accepted__min__ALL_TIME', 'agg_payments__g_contract__sum__all__countDistinct__1W', 'agg_payments__g_contract__sum__all__countDistinct__2W', 'agg_payments__g_contract__sum__all__countDistinct__4W', 'agg_payments__g_contract__sum__all__countDistinct__8W', 'agg_payments__g_contract__sum__all__countDistinct__12W', 'agg_payments__g_contract__sum__all__countDistinct__ALL_TIME', 'agg_payments__g_contract__sum__all__sum__1W', 'agg_payments__g_contract__sum__all__sum__2W', 'agg_payments__g_contract__sum__all__sum__4W', 'agg_payments__g_contract__sum__all__sum__8W', 'agg_payments__g_contract__sum__all__sum__12W', 'agg_payments__g_contract__sum__all__sum__ALL_TIME', 'counteragent_mean_agg_payments__g_contract__sum__all__sum__ALL_TIME', 'counteragent_sum_agg_payments__g_contract__sum__all__sum__ALL_TIME', 'specialization_sum_agg_payments__g_contract__sum__all__sum__ALL_TIME', 'agg_payments__g_contract__date__advance__min__ALL_TIME', 'agg_ks2__g_contract__id__all__count__1W', 'agg_ks2__g_contract__id__all__count__2W', 'agg_ks2__g_contract__id__all__count__4W', 'agg_ks2__g_contract__id__all__count__8W', 'agg_ks2__g_contract__id__all__count__12W', 'agg_ks2__g_contract__id__all__count__ALL_TIME', 'agg_ks2__g_contract__total_sum__all__sum__1W', 'agg_ks2__g_contract__total_sum__all__sum__2W', 'agg_ks2__g_contract__total_sum__all__sum__4W', 'agg_ks2__g_contract__total_sum__all__sum__8W', 'agg_ks2__g_contract__total_sum__all__sum__12W', 'agg_ks2__g_contract__total_sum__all__sum__ALL_TIME', 'counteragent_mean_agg_ks2__g_contract__total_sum__all__sum__ALL_TIME', 'counteragent_sum_agg_ks2__g_contract__total_sum__all__sum__ALL_TIME', 'specialization_sum_agg_ks2__g_contract__total_sum__all__sum__ALL_TIME', 'agg_spass_applications__g_contract__appl_count_week__max__ALL_TIME', 'agg_spass_applications__g_contract__appl_count_week__mean__1W', 'agg_spass_applications__g_contract__appl_count_week__mean__2W', 'agg_spass_applications__g_contract__appl_count_week__mean__3W', 'agg_spass_applications__g_contract__appl_count_week__mean__4W', 'agg_spass_applications__g_contract__appl_count_week__mean__5W', 'agg_spass_applications__g_contract__appl_count_week__mean__6W', 'agg_spass_applications__g_contract__appl_count_week__mean__8W', 'agg_spass_applications__g_contract__appl_count_week__mean__12W', 'agg_spass_applications__g_contract__appl_count_week__mean__26W', 'agg_spass_applications__g_contract__appl_count_week__mean__ALL_TIME', 'agg_spass_applications__g_specialization__appl_count_week__mean__ALL_TIME', 'counteragent_mean_agg_spass_applications__g_contract__appl_count_week__mean__ALL_TIME', 'agg_workers__g_contract__fact_workers__all__mean__1W', 'agg_workers__g_contract__fact_workers__all__mean__2W', 'agg_workers__g_contract__fact_workers__all__mean__3W', 'agg_workers__g_contract__fact_workers__all__mean__4W', 'agg_workers__g_contract__fact_workers__all__mean__5W', 'agg_workers__g_contract__fact_workers__all__mean__6W', 'agg_workers__g_contract__fact_workers__all__mean__8W', 'agg_workers__g_contract__fact_workers__all__mean__12W', 'agg_workers__g_contract__fact_workers__all__mean__26W', 'agg_workers__g_contract__fact_workers__all__mean__ALL_TIME', 'agg_materials__g_contract__order_id__countDistinct__1W', 'agg_materials__g_contract__order_id__countDistinct__2W', 'agg_materials__g_contract__order_id__countDistinct__4W', 'agg_materials__g_contract__order_id__countDistinct__8W', 'agg_materials__g_contract__order_id__countDistinct__12W', 'agg_materials__g_contract__order_id__countDistinct__ALL_TIME', 'agg_materials__g_contract__order_dt__min__ALL_TIME', 'agg_materials__g_contract__material_type_id__countDistinct__ALL_TIME', 'agg_materials__g_contract__material_id__countDistinct__ALL_TIME', 'agg_scontrol__g_contractor__close_delay__defect_type_repair__mean__ALL_TIME', 'agg_scontrol__g_contractor__close_delay__defect_type_author_supervision__mean__ALL_TIME', 'agg_scontrol__g_contractor__close_delay__defect_type_GR__mean__ALL_TIME', 'agg_scontrol__g_contractor__close_delay__defect_type_labour_protection__mean__ALL_TIME', 'agg_scontrol__g_contractor__close_delay__defect_type_general_contractor__mean__ALL_TIME', 'agg_scontrol__g_contractor__close_delay__defect_type_arch_supervision__mean__ALL_TIME', 'agg_scontrol__g_contractor__close_delay__defect_type_tech_supervision__mean__ALL_TIME', 'agg_scontrol__g_contractor__close_delay__defect_type_app__mean__ALL_TIME', 'agg_sroomer__g_contractor__sroomer_id__count__3M', 'agg_sroomer__g_contractor__sroomer_id__count__6M', 'agg_sroomer__g_contractor__sroomer_id__count__12M', 'agg_sroomer__g_contractor__sroomer_id__count__ALL_TIME', 'agg_BoardOfDirectors__g_contractor__Name__count__ALL_TIME', 'agg_ConsolidatedIndicator__g_contractor__Index__Overall__mean__ALL_TIME', 'agg_ConsolidatedIndicator__g_contractor__Index__FailureScore__mean__ALL_TIME', 'agg_ConsolidatedIndicator__g_contractor__Index__PaymentIndex__mean__ALL_TIME', 'agg_ConsolidatedIndicator__g_contractor__Index__IndexOfDueDiligence__mean__ALL_TIME', 'agg_spark_extended_report__g_contractor__EstimatedClaimsSum__last__ALL_TIME', 'agg_spark_extended_report__g_contractor__EstimatedLiabilitiesSum__last__ALL_TIME', 'agg_spark_extended_report__g_contractor__EstimatedNetLiabilitiesSum__last__ALL_TIME', 'agg_spark_extended_report__g_contractor__PledgeeActiveCount__last__ALL_TIME', 'agg_spark_extended_report__g_contractor__PledgeeCeasedCount__last__ALL_TIME', 'agg_spark_extended_report__g_contractor__PledgerActiveCount__last__ALL_TIME', 'agg_spark_extended_report__g_contractor__PledgerCeasedCount__last__ALL_TIME', 'agg_spark_extended_report__g_contractor__CompanySizeRevenue__last__ALL_TIME', 'agg_spark_extended_report__g_contractor__CreditLimitSum__last__ALL_TIME', 'agg_Finance__g_contractor__Value__FixedAssets__last__ALL_TIME', 'agg_Finance__g_contractor__Value__CurrentAssets__last__ALL_TIME', 'agg_Finance__g_contractor__Value__Capital__last__ALL_TIME', 'agg_Finance__g_contractor__Value__LongLiabilities__last__ALL_TIME', 'agg_Finance__g_contractor__Value__ShortLiabilities__last__ALL_TIME', 'agg_Finance__g_contractor__Value__Balance__last__ALL_TIME', 'agg_Finance__g_contractor__Value__Revenue_y__last__ALL_TIME', 'agg_Finance__g_contractor__Value__NetProfit_y__last__ALL_TIME', 'agg_Finance__g_contractor__Value__CostPrice_y__last__ALL_TIME', 'agg_FinanceAndTaxesFTS__g_contractor__Expenses__last__ALL_TIME', 'agg_FinanceAndTaxesFTS__g_contractor__Income__last__ALL_TIME', 'agg_FinanceAndTaxesFTS__g_contractor__TaxArrearsSum__last__ALL_TIME', 'agg_FinanceAndTaxesFTS__g_contractor__TaxPenaltiesSum__last__ALL_TIME', 'agg_FinanceAndTaxesFTS__g_contractor__TaxesSum__last__ALL_TIME', 'agg_ArbitrationCases__g_contractor__DefendantSum__sum__12M', 'agg_ArbitrationCases__g_contractor__DefendantSum__sum__12_24M', 'agg_ArbitrationCases__g_contractor__DefendantSum__sum__12_36M', 'agg_ArbitrationCases__g_contractor__DefendantSum__sum__12_48M', 'agg_ArbitrationCases__g_contractor__DefendantSum__sum__ALL_TIME', 'agg_ArbitrationCases__g_contractor__PlaintiffSum__sum__12M', 'agg_ArbitrationCases__g_contractor__PlaintiffSum__sum__12_24M', 'agg_ArbitrationCases__g_contractor__PlaintiffSum__sum__12_36M', 'agg_ArbitrationCases__g_contractor__PlaintiffSum__sum__12_48M', 'agg_ArbitrationCases__g_contractor__PlaintiffSum__sum__ALL_TIME', 'agg_tender_proposal__g_contractor__id__ALL__countDistinct__1W', 'agg_tender_proposal__g_contractor__id__ALL__countDistinct__2W', 'agg_tender_proposal__g_contractor__id__ALL__countDistinct__4W', 'agg_tender_proposal__g_contractor__id__ALL__countDistinct__8W', 'agg_tender_proposal__g_contractor__id__ALL__countDistinct__12W', 'agg_tender_proposal__g_contractor__id__ALL__countDistinct__26W', 'agg_tender_proposal__g_contractor__id__ALL__countDistinct__52W', 'agg_tender_proposal__g_contractor__id__ALL__countDistinct__ALL_TIME']

import pandas as pd
import numpy as np
from datetime import date, datetime
from catboost import CatBoostClassifier, Pool as CatBoostPool
import os
from . import config
from .scoringModel import ScoringModel

class ProcessingInputInvalid(Exception):
	pass

class CatBoostModel:
	def __init__(self):
		self.cat_features = ['project_id', 'building_id', 'contractor_id', 'report_month', 'report_day', 'contract_year', 'contract_month']
		cb_path = config.get_settings().catboost_model_path
		self.model = CatBoostClassifier()
		self.model.load_model(cb_path)
	
	def normalize_Xdf(self, df: pd.DataFrame) -> pd.DataFrame:
		"""
		Prepares the data frame to be used by the model.
		"""
		df = df.copy()

		df['report_date'] = df['report_date'].map(lambda x: date.fromisoformat(x))
		df['report_month'] = df['report_date'].map(lambda x: x.month)
		df['report_day'] = df['report_date'].map(lambda x: x.day)

		df['contract_date'] = df['contract_date'].map(lambda x: datetime.fromisoformat(x))
		df['contract_year'] = df['contract_date'].map(lambda x: x.year)
		df['contract_month'] = df['contract_date'].map(lambda x: x.month)

		df = df.drop(columns=['report_date', 'contract_date'])
		df = df.fillna(0.)

		return df

	def predict(self, Xdf: pd.DataFrame) -> list[float]:
			
		ndf = self.normalize_Xdf(Xdf).drop(columns=["contract_id"])
		pool = CatBoostPool(ndf, cat_features=self.cat_features)
		res = self.model.predict_proba(pool)[:,1]
		return res.tolist()
	
	def __call__(self, Xdf: pd.DataFrame) -> list[float]:
		return self.predict(Xdf)

class Model:
	"""
	Processes the normalized data frame (after DataPreprocessor stage) and
	return the target output
	"""
	def __init__(self):
		self.model = ScoringModel()
	
	def predict(self, Xdf: pd.DataFrame) -> list[float]:
		"""
		Makes prediction and returns the result.
		"""
		return self.model.predict_scoring(Xdf).tolist()

	def __call__(self, Xdf: pd.DataFrame) -> list[float]:
		return self.predict(Xdf)

class Predictor():
	"""
	Pass user input data frame to this class and get output results.
	Basic class for model processing. Only this class should be used
	by API.
	"""
	
	def __init__(self):
		self.model = Model()

	def predict(self, input_df: pd.DataFrame) -> list[float]:
		if not self.validate_input(input_df):
			raise ProcessingInputInvalid()
			
		outputs = self.model(input_df)
		
		return outputs

	def validate_input(self, input_df: pd.DataFrame) -> bool:
		if not np.array_equal(input_df.columns, DATA_ALLOWED_COLUMNS):
			return False

		return True
	
