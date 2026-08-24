import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')


class OLSLogisticTool:
    """
    OLS / Logistic 回归因果推断工具类
    支持连续型和二元处理变量的因果效应估计
    """
    
    main_color = "#4C72B0"
    line_color = "#8A8A8A"
    
    def __init__(self, file_path, treatment_col, outcome_col, control_cols, 
                 model_type='ols', save_dir=None):
        """
        Parameters:
        -----------
        file_path : str
            数据文件路径
        treatment_col : str
            处理变量列名
        outcome_col : str
            结果变量列名
        control_cols : list
            控制变量列名列表
        model_type : str
            模型类型: 'ols' (线性回归) 或 'logistic' (逻辑回归)
        save_dir : str, optional
            保存目录
        """
        self.file_path = file_path
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.control_cols = control_cols
        self.model_type = model_type.lower()
        
        if save_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            self.save_dir = os.path.join(project_root, 'data_analysis_result/causal_analysis')
        else:
            self.save_dir = save_dir
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        self._setup_plot_style()
        
        # 读取数据
        self.df = pd.read_excel(file_path)
        
        # 验证列
        all_cols = [treatment_col, outcome_col] + control_cols
        missing_cols = [col for col in all_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"以下列不存在于数据中: {missing_cols}")
        
        self.T = self.df[treatment_col].values
        self.Y = self.df[outcome_col].values
        self.W = self.df[control_cols].values
        
        self.feature_names = control_cols
        self.treatment_name = treatment_col
        self.outcome_name = outcome_col
        
        # 结果存储
        self.coefficients = None
        self.p_values = None
        self.conf_int = None
        self.r_squared = None
        self.model = None
    
    def _setup_plot_style(self):
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 0.8,
            "figure.dpi": 300
        })
        sns.set_style("white")
    
    def run_estimation(self):
        """执行回归估计"""
        print("=" * 60)
        print(f"开始 {self.model_type.upper()} 回归因果效应估计...")
        print("=" * 60)
        print(f"处理变量: {self.treatment_name}")
        print(f"结果变量: {self.outcome_name}")
        print(f"控制变量: {', '.join(self.control_cols)}")
        print(f"样本量: {len(self.df)}")
        print("-" * 60)
        
        # 准备数据
        X = np.column_stack([self.T.reshape(-1, 1), self.W])
        X_with_const = sm.add_constant(X)
        
        # 构建模型
        if self.model_type == 'ols':
            self.model = sm.OLS(self.Y, X_with_const).fit()
        elif self.model_type == 'logistic':
            # 对于逻辑回归，结果变量应该是二元的
            if not np.all(np.isin(self.Y, [0, 1])):
                print("Warning: Outcome variable is not binary. Consider using OLS instead.")
            self.model = sm.Logit(self.Y, X_with_const).fit(disp=0)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # 提取结果
        self.coefficients = self.model.params
        self.p_values = self.model.pvalues
        self.conf_int = self.model.conf_int()
        
        # 确保 conf_int 是 DataFrame
        if isinstance(self.conf_int, np.ndarray):
            var_names = ['const'] + [self.treatment_name] + self.control_cols
            self.conf_int = pd.DataFrame(self.conf_int, index=var_names, columns=['CI Lower', 'CI Upper'])
        
        # 确保 coefficients 和 p_values 是 numpy 数组（如果是 Series 则转换）
        if hasattr(self.coefficients, 'values'):
            self.coefficients = self.coefficients.values
        if hasattr(self.p_values, 'values'):
            self.p_values = self.p_values.values
        
        if self.model_type == 'ols':
            self.r_squared = self.model.rsquared
        
        # 打印结果
        print("\n✅ 估计完成！")
        print("\n==== 回归结果 ====")
        print(f"\n{self.model.summary().tables[1]}")
        
        # 处理效应
        treatment_idx = 1  # Treatment 是第二个变量（第一个是常数项）
        ate = self.coefficients[treatment_idx]
        p_value = self.p_values[treatment_idx]
        ci_lower = self.conf_int.iloc[treatment_idx, 0]
        ci_upper = self.conf_int.iloc[treatment_idx, 1]
        
        print(f"\n==== 因果效应估计 ====")
        print(f"平均处理效应 (ATE): {ate:.4f}")
        print(f"95% 置信区间: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"P值: {p_value:.4f}")
        
        if p_value < 0.01:
            print("显著性: *** (p < 0.01)")
        elif p_value < 0.05:
            print("显著性: ** (p < 0.05)")
        elif p_value < 0.1:
            print("显著性: * (p < 0.1)")
        else:
            print("显著性: 不显著")
        
        if self.model_type == 'ols':
            print(f"\nR-squared: {self.r_squared:.4f}")
        
        return ate
    
    def plot_coefficients(self, output_file=None, save_dir=None):
        """绘制系数图"""
        if output_file is None:
            output_file = f"{self.model_type}_{self.treatment_name}_coefficients.png"
        
        if save_dir is None:
            save_dir = self.save_dir
        elif not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        full_path = os.path.join(save_dir, output_file)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 准备数据
        var_names = ['const'] + [self.treatment_name] + self.control_cols
        # coefficients 已经是 numpy 数组了，不需要 .values
        coefs = self.coefficients if isinstance(self.coefficients, np.ndarray) else np.array(self.coefficients)
        ci_lower = self.conf_int.iloc[:, 0].values
        ci_upper = self.conf_int.iloc[:, 1].values
        
        # 排序（按系数大小）
        sorted_idx = np.argsort(coefs)
        var_names = [var_names[i] for i in sorted_idx]
        coefs = coefs[sorted_idx]
        ci_lower = ci_lower[sorted_idx]
        ci_upper = ci_upper[sorted_idx]
        
        y_pos = np.arange(len(var_names))
        
        # 绘制系数和置信区间
        colors = [self.main_color if name == self.treatment_name else '#95A5A6' for name in var_names]
        ax.barh(y_pos, coefs, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.errorbar(coefs, y_pos, xerr=[coefs - ci_lower, ci_upper - coefs], 
                   fmt='none', color='black', capsize=3, linewidth=1)
        
        # 添加零线
        ax.axvline(x=0, color=self.line_color, linestyle='--', linewidth=1)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(var_names, fontsize=10)
        ax.set_xlabel('Coefficient', fontsize=12)
        ax.set_title(f'Regression Coefficients ({self.model_type.upper()})', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(full_path, dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 已保存系数图：{full_path}")
        return full_path
    
    def save_results(self, output_file=None, save_dir=None):
        """保存结果到 Excel"""
        if output_file is None:
            output_file = f"{self.model_type}_{self.treatment_name}_results.xlsx"
        
        if save_dir is None:
            save_dir = self.save_dir
        elif not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        full_path = os.path.join(save_dir, output_file)
        
        # 系数结果
        # coefficients 和 p_values 已经是 numpy 数组了
        coef_data = {
            'Variable': ['const'] + [self.treatment_name] + self.control_cols,
            'Coefficient': self.coefficients if isinstance(self.coefficients, np.ndarray) else np.array(self.coefficients),
            'Std Error': self.model.bse.values if hasattr(self.model.bse, 'values') else np.array(self.model.bse),
            'P-value': self.p_values if isinstance(self.p_values, np.ndarray) else np.array(self.p_values),
            'CI Lower': self.conf_int.iloc[:, 0].values,
            'CI Upper': self.conf_int.iloc[:, 1].values
        }
        coef_df = pd.DataFrame(coef_data)
        
        # 模型信息
        model_info = pd.DataFrame({
            'Parameter': ['Treatment Variable', 'Outcome Variable', 'Control Variables', 
                         'Sample Size', 'Method', 'Model Type'],
            'Value': [
                self.treatment_name,
                self.outcome_name,
                ', '.join(self.control_cols),
                len(self.df),
                'Regression-based Causal Inference',
                self.model_type.upper()
            ]
        })
        
        if self.model_type == 'ols':
            model_info = pd.concat([model_info, pd.DataFrame({
                'Parameter': ['R-squared', 'Adj R-squared'],
                'Value': [self.model.rsquared, self.model.rsquared_adj]
            })], ignore_index=True)
        
        with pd.ExcelWriter(full_path, engine='openpyxl') as writer:
            coef_df.to_excel(writer, sheet_name='Coefficients', index=False)
            model_info.to_excel(writer, sheet_name='Model_Info', index=False)
        
        print(f"💾 已保存结果：{full_path}")
        return full_path
    
    def run_full_analysis(self, save_dir=None):
        """运行完整分析"""
        if save_dir:
            self.save_dir = save_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        
        self.run_estimation()
        self.save_results()
        self.plot_coefficients()
        
        print("\n✅ OLS/Logistic 完整分析完成！")


if __name__ == "__main__":
    print("OLS/Logistic 因果推断工具")
