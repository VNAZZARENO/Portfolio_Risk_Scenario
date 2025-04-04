import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import statsmodels.api as sm

class FactorMonitor:
    def __init__(self, data_file, fund_column='PERGLFI FP Equity'):
        """
        Initialize the FactorMonitor with data and factor groupings
        
        Parameters:
        data_file (str): Path to the CSV file containing all data
        fund_column (str): Name of the column containing fund data
        """
        self.data = pd.read_csv(data_file, sep=";")
        self.data['Dates'] = pd.to_datetime(self.data['Dates'], format="%d/%m/%Y")
        self.data.set_index('Dates', inplace=True)
        
        self.fund_column = fund_column
        
        self.factor_groups = {
            'rates': [
                'H15T3M Index', 'H15T1Y Index', 'H15T5Y Index', 'H15T10Y Index',
                'GTDEM10Y Govt', 'CTDEM10Y Govt', 'CTFRF10Y Govt', 'CTDEM10Y Govt.1',
                'CTITL10Y Govt', 'CTEUR10Y Govt', 'GUKG30 Index', 'CTEUR7Y Govt',
                'JPEI3MEU Index', 'GDP CURY Index', 'OATA Comdty'
            ],
            'credit': [
                'FRANCE CDS USD SR 10Y D14 Corp', 'UK CDS USD SR 10Y D14 Corp',
                'SPAIN CDS USD SR 10Y D14 Corp', 'ITALY CDS USD SR 10Y D14 Corp',
                'GERMAN CDS USD SR 10Y D14 Corp', 'LF98TRUU Index', 'IBXXCHF3 Index'
            ],
            'fx': ['EURUSD Curncy', 'EURGBP Curncy'],
            'equity': ['ESES Index', 'ASDA Index', 'DEDA Index'],
            'commodities': ['XAU Curncy', 'ERA Comdty', 'CLA Comdty'],
            'volatility': ['VGM5 Index', 'VGA Index']
        }
        
        self.exposure_results = {}
        self.attribution_results = {}
        self.risk_decomposition = {}
        self.scenario_results = {}
    
    def calculate_factor_exposures(self, lookback_period=60):
        """
        Calculate exposures to various factor groups based on regression analysis
        
        Parameters:
        lookback_period (int): Number of days to use for regression analysis
        """
        recent_data = self.data.tail(lookback_period)
        
        for group_name, factors in self.factor_groups.items():
            group_exposures = {}
            
            for factor in factors:
                if factor in recent_data.columns:
                    exposure = self._calculate_sensitivity_to_factor(recent_data, factor)
                    group_exposures[factor] = exposure
            
            self.exposure_results[group_name] = group_exposures
            
        return self.exposure_results
    
    def _calculate_sensitivity_to_factor(self, data, factor):
        """
        Calculate the sensitivity of the fund to a specific factor using regression
        
        Parameters:
        data (DataFrame): DataFrame containing the data
        factor (str): Name of the factor
        
        Returns:
        float: Sensitivity (beta) to the factor
        """
        if factor not in data.columns or self.fund_column not in data.columns:
            return 0
            
        y = data[self.fund_column].pct_change().dropna()
        x = data[factor].pct_change().dropna()
        
        x, y = x.align(y, join='inner')
        
        if len(x) < 10:  
            return 0
            
        X = pd.DataFrame({factor: x})
        X = sm.add_constant(X)
        
        try:
            model = sm.OLS(y, X).fit()
            beta = model.params[factor]
            return beta
        except:
            return 0
    
    def perform_return_attribution(self, start_date, end_date):
        """
        Attribution analysis for a specific period
        
        Parameters:
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        """
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        period_data = self.data.loc[start_date:end_date]
        fund_return = period_data[self.fund_column].iloc[-1] / period_data[self.fund_column].iloc[0] - 1
        
        attribution = {}
        unexplained = fund_return
        
        for group_name, factors in self.factor_groups.items():
            group_contribution = 0
            for factor in factors:
                if factor in period_data.columns:
                    factor_return = period_data[factor].iloc[-1] / period_data[factor].iloc[0] - 1
                    
                    if group_name in self.exposure_results and factor in self.exposure_results[group_name]:
                        exposure = self.exposure_results[group_name][factor]
                        contribution = exposure * factor_return
                        group_contribution += contribution
                        unexplained -= contribution
            
            attribution[group_name] = group_contribution
        
        attribution['unexplained'] = unexplained
        self.attribution_results = attribution
        return attribution
    
    def decompose_risk(self, lookback_period=60):
        """
        Decompose portfolio risk by factor groups
        
        Parameters:
        lookback_period (int): Number of days to use for risk calculation
        """
        recent_data = self.data.tail(lookback_period)
        returns = recent_data.pct_change().dropna()
        
        
        cov_matrix = returns.cov()
        fund_vol = returns[self.fund_column].std() * np.sqrt(252)  # Annualized
        
        risk_decomp = {}
        total_explained_risk = 0
        
        for group_name, factors in self.factor_groups.items():
            group_risk = 0
            valid_factors = [f for f in factors if f in returns.columns and 
                            group_name in self.exposure_results and 
                            f in self.exposure_results[group_name]]
            
            for i, factor1 in enumerate(valid_factors):
                beta1 = self.exposure_results[group_name][factor1]
                
                # Factor risk contribution = beta * volatility * correlation with fund
                if factor1 in cov_matrix.columns and self.fund_column in cov_matrix.columns:
                    factor_vol = returns[factor1].std() * np.sqrt(252)  # Annualized
                    factor_contrib = beta1 * factor_vol
                    group_risk += factor_contrib
            
            risk_decomp[group_name] = abs(group_risk)
            total_explained_risk += abs(group_risk)
            
        unexplained_risk = fund_vol - total_explained_risk
        if unexplained_risk > 0:
            risk_decomp['unexplained'] = unexplained_risk
            total_explained_risk += unexplained_risk
            
        for group in risk_decomp:
            risk_decomp[group] = risk_decomp[group] / total_explained_risk * 100
            
        self.risk_decomposition = risk_decomp
        return risk_decomp
    
    def run_scenario_analysis(self, scenarios):
        """
        Run scenario analysis on the portfolio
        
        Parameters:
        scenarios (dict): Dictionary of scenarios with factor shocks
        
        Example:
        scenarios = {
            'rates_up_100bp': {'H15T10Y Index': 1.0, 'GTDEM10Y Govt': 1.0},
            'equity_down_10pct': {'ESES Index': -0.1},
        }
        """
        scenario_results = {}
        
        for scenario_name, scenario_shocks in scenarios.items():
            scenario_impact = 0
            
            for factor, shock in scenario_shocks.items():
                for group_name, factors in self.factor_groups.items():
                    if factor in factors and group_name in self.exposure_results and factor in self.exposure_results[group_name]:
                        exposure = self.exposure_results[group_name][factor]
                        impact = exposure * shock * 100  
                        scenario_impact += impact
                        break
            
            scenario_results[scenario_name] = scenario_impact
            
        self.scenario_results = scenario_results
        return scenario_results
    
    def generate_report(self, output_file):
        """
        Generate a report with all the analysis results
        
        Parameters:
        output_file (str): Path to save the Excel report
        """
        with pd.ExcelWriter(output_file) as writer:
            # Factor exposures
            exposure_df = pd.DataFrame()
            for group, exposures in self.exposure_results.items():
                if exposures:  # Skip empty groups
                    group_df = pd.DataFrame.from_dict(exposures, orient='index', columns=[group])
                    exposure_df = pd.concat([exposure_df, group_df], axis=1)
            
            if not exposure_df.empty:
                exposure_df.to_excel(writer, sheet_name='Factor Exposures')
            
            if self.attribution_results:
                attr_df = pd.DataFrame.from_dict(self.attribution_results, orient='index', columns=['Contribution'])
                attr_df.to_excel(writer, sheet_name='Return Attribution')
            
            if self.risk_decomposition:
                risk_df = pd.DataFrame.from_dict(self.risk_decomposition, orient='index', columns=['Risk Contribution (%)'])
                risk_df.to_excel(writer, sheet_name='Risk Decomposition')
            
            if self.scenario_results:
                scenario_df = pd.DataFrame.from_dict(self.scenario_results, orient='index', columns=['Impact (%)'])
                scenario_df.to_excel(writer, sheet_name='Scenario Analysis')
            
        self._generate_charts(output_file.replace('.xlsx', '_charts.pdf'))
    
    def _generate_charts(self, output_file):
        """
        Generate charts for visualization
        
        Parameters:
        output_file (str): Path to save the PDF with charts
        """
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        if self.exposure_results:
            exposure_data = {}
            for group, exposures in self.exposure_results.items():
                exposure_data[group] = sum(exposures.values()) if exposures else 0
            
            axs[0, 0].bar(exposure_data.keys(), exposure_data.values())
            axs[0, 0].set_title('Net Factor Exposures')
            axs[0, 0].tick_params(axis='x', rotation=45)
        
        if self.attribution_results:
            axs[0, 1].bar(self.attribution_results.keys(), self.attribution_results.values())
            axs[0, 1].set_title('Return Attribution')
            axs[0, 1].tick_params(axis='x', rotation=45)
        
        if self.risk_decomposition:
            axs[1, 0].pie(self.risk_decomposition.values(), labels=self.risk_decomposition.keys(), autopct='%1.1f%%')
            axs[1, 0].set_title('Risk Decomposition')
        
        if self.scenario_results:
            axs[1, 1].bar(self.scenario_results.keys(), self.scenario_results.values())
            axs[1, 1].set_title('Scenario Analysis Impact')
            axs[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_file)
        
    def run_full_analysis(self, start_date, end_date, lookback_period=60, scenarios=None):
        """
        Run all analyses in sequence
        
        Parameters:
        start_date (str): Start date for attribution in format 'YYYY-MM-DD'
        end_date (str): End date for attribution in format 'YYYY-MM-DD'
        lookback_period (int): Number of days for exposure and risk calculation
        scenarios (dict): Dictionary of scenarios with factor shocks
        """
        import statsmodels.api as sm
        
        print("Calculating factor exposures...")
        self.calculate_factor_exposures(lookback_period)
        
        print("Performing return attribution...")
        self.perform_return_attribution(start_date, end_date)
        
        print("Decomposing risk...")
        self.decompose_risk(lookback_period)
        
        if scenarios:
            print("Running scenario analysis...")
            self.run_scenario_analysis(scenarios)
        
        return {
            "exposures": self.exposure_results,
            "attribution": self.attribution_results,
            "risk": self.risk_decomposition,
            "scenarios": self.scenario_results
        }
    
import statsmodels.api as sm

data_file = "PATH/TO/FILE.CSV"

monitor = FactorMonitor(data_file)

start_date = '2020-01-01'
end_date = '2025-04-04'

scenarios = {
    'rates_up_100bp': {
        'H15T10Y Index': 1.0,
        'GTDEM10Y Govt': 1.0,
        'CTEUR10Y Govt': 1.0
    },
    'equity_down_10pct': {
        'ESES Index': -0.1,
        'ASDA Index': -0.1,
        'DEDA Index': -0.1
    },
    'credit_widening': {
        'FRANCE CDS USD SR 10Y D14 Corp': 0.5,
        'ITALY CDS USD SR 10Y D14 Corp': 0.5,
        'SPAIN CDS USD SR 10Y D14 Corp': 0.5
    },
    'fx_euro_strengthening': {
        'EURUSD Curncy': 0.05,
        'EURGBP Curncy': 0.05
    }
}

results = monitor.run_full_analysis(start_date, end_date, scenarios=scenarios)
monitor.generate_report("factor_analysis_report.xlsx")
