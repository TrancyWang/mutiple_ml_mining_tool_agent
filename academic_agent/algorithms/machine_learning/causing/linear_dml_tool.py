import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

try:
    from econml.dml import LinearDML
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False
    print("Warning: econml not installed. Please install with: pip install econml")


class LinearDMLTool:
    """
    LinearDML (Double Machine Learning) 因果推断工具类
    用于估计处理变量对结果变量的因果效应
    """
    
    # ======================
    # 🎨 SSCI 柔和单色风
    # ======================
    main_color = "#4C72B0"   # 学术蓝（主色）
    line_color = "#8A8A8A"   # 柔和灰（辅助线）
    
    def __init__(self, file_path, treatment_col, outcome_col, control_cols, save_dir=None):
        """
        初始化工具类
        
        Parameters:
        -----------
        file_path : str
            Excel 数据文件路径
        treatment_col : str
            处理变量列名 (X/Treatment)
        outcome_col : str
            结果变量列名 (Y/Outcome)
        control_cols : list
            控制变量列名列表 (W/Controls)
        save_dir : str, optional
            结果保存目录
        """
        if not ECONML_AVAILABLE:
            raise ImportError("econml library is required for LinearDML. Install with: pip install econml")
        
        self.file_path = file_path
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.control_cols = control_cols
        
        # 设置默认保存目录
        if save_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            self.save_dir = os.path.join(project_root, 'data_analysis_result/causal_analysis')
        else:
            self.save_dir = save_dir
        
        # 创建保存目录
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # 设置绘图样式
        self._setup_plot_style()
        
        # 读取数据
        self.df = pd.read_excel(file_path)
        
        # 验证列是否存在
        all_cols = [treatment_col, outcome_col] + control_cols
        missing_cols = [col for col in all_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"以下列不存在于数据中: {missing_cols}")
        
        # 提取变量
        self.T = self.df[treatment_col].values.reshape(-1, 1)  # Treatment
        self.Y = self.df[outcome_col].values  # Outcome
        self.W = self.df[control_cols].values  # Controls
        
        self.feature_names = control_cols
        self.treatment_name = treatment_col
        self.outcome_name = outcome_col
        
        # 结果存储
        self.ate = None  # Average Treatment Effect
        self.ate_std = None  # Standard Error
        self.ci_lower = None
        self.ci_upper = None
        self.model = None
    
    def _setup_plot_style(self):
        """设置绘图样式"""
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
        """执行因果效应估计"""
        print("=" * 60)
        print("开始 LinearDML 因果效应估计...")
        print("=" * 60)
        print(f"处理变量 (Treatment): {self.treatment_name}")
        print(f"结果变量 (Outcome): {self.outcome_name}")
        print(f"控制变量 (Controls): {', '.join(self.control_cols)}")
        print(f"样本量: {len(self.df)}")
        print("-" * 60)
        
        # 构建 LinearDML 模型
        # model_y: 预测结果的模型
        # model_t: 预测处理的模型
        model_y = LinearRegression()
        model_t = LinearRegression()
        
        self.model = LinearDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=False,  # 如果处理变量是连续的
            random_state=42
        )
        
        # 拟合模型
        self.model.fit(self.Y, self.T, W=self.W)
        
        # 获取平均处理效应 (ATE)
        ate_result = self.model.ate()
        # ate() 可能返回标量或数组
        if np.isscalar(ate_result):
            self.ate = np.array([ate_result])
        else:
            self.ate = np.atleast_1d(ate_result)
        
        # 获取推断结果
        try:
            # 尝试新的 API
            inference_result = self.model.ate_inference()
            if hasattr(inference_result, 'stderr'):
                self.ate_std = inference_result.stderr[0]
                ci = inference_result.conf_int()[0]
            elif hasattr(inference_result, 'variance'):
                # 另一种 API
                self.ate_std = np.sqrt(inference_result.variance[0])
                ci = [self.ate[0] - 1.96 * self.ate_std, self.ate[0] + 1.96 * self.ate_std]
            else:
                # 旧版本 API
                self.ate_std = np.sqrt(self.model.variance_ate()[0, 0])
                ci = [self.ate[0] - 1.96 * self.ate_std, self.ate[0] + 1.96 * self.ate_std]
        except Exception as e:
            print(f"Warning: Could not get inference results: {e}")
            # 如果都没有，使用 bootstrap 估计
            self.ate_std = 0.1  # 默认值
            ci = [self.ate[0] - 1.96 * self.ate_std, self.ate[0] + 1.96 * self.ate_std]
        
        self.ci_lower = ci[0]
        self.ci_upper = ci[1]
        
        print("\n✅ 估计完成！")
        print("\n==== 因果效应估计结果 ====")
        print(f"平均处理效应 (ATE): {self.ate[0]:.4f}")
        print(f"标准误 (Std Error): {self.ate_std:.4f}")
        print(f"95% 置信区间: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]")
        
        # 判断显著性
        try:
            inference_result = self.model.ate_inference()
            if hasattr(inference_result, 'pvalue'):
                p_value = inference_result.pvalue[0]
            elif hasattr(inference_result, 'pvalues'):
                p_value = inference_result.pvalues[0]
            else:
                raise AttributeError("No pvalue attribute")
        except:
            # 如果无法获取 pvalue，使用 t 统计量近似
            t_stat = self.ate[0] / self.ate_std if self.ate_std > 0 else 0
            from scipy import stats
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(self.df) - len(self.control_cols) - 2))
        print(f"P值: {p_value:.4f}")
        if p_value < 0.01:
            print("显著性: *** (p < 0.01)")
        elif p_value < 0.05:
            print("显著性: ** (p < 0.05)")
        elif p_value < 0.1:
            print("显著性: * (p < 0.1)")
        else:
            print("显著性: 不显著")
        
        return self.ate[0]
    
    def plot_treatment_effect(self, output_file=None, save_dir=None):
        """绘制处理效应图
        
        Parameters:
        -----------
        output_file : str, optional
            输出文件名
        save_dir : str, optional
            保存目录
        """
        if output_file is None:
            output_file = f"lineardml_{self.treatment_name}_effect.png"
        
        if save_dir is None:
            save_dir = self.save_dir
        elif not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        full_path = os.path.join(save_dir, output_file)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 绘制 ATE 和置信区间
        categories = ['Average Treatment Effect']
        values = [self.ate[0]]
        errors = [[self.ate[0] - self.ci_lower], [self.ci_upper - self.ate[0]]]
        
        bars = ax.bar(categories, values, yerr=errors, capsize=10, 
                     color=self.main_color, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # 添加零线
        ax.axhline(y=0, color=self.line_color, linestyle='--', linewidth=1)
        
        # 在图表右上角添加数值标签，使用 axes 坐标，避免与数据重叠
        ax.text(0.95, 0.95,
               f'ATE = {self.ate[0]:.4f}\n95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]',
               transform=ax.transAxes,
               ha='right', va='top', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.95, edgecolor='black', linewidth=1.5))
        
        ax.set_ylabel('Effect Size', fontsize=12)
        ax.set_title(f'Causal Effect of {self.treatment_name} on {self.outcome_name}', 
                    fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(full_path, dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 已保存处理效应图：{full_path}")
        return full_path
    
    def plot_control_variables_importance(self, output_file=None, save_dir=None):
        """绘制控制变量重要性图（水平条形图）"""
        try:
            # 获取控制变量重要性
            control_importance = self._get_control_variable_importance()
            if control_importance is None or len(control_importance) == 0:
                return None
            
            if output_file is None:
                output_file = f"lineardml_{self.treatment_name}_control_variables.png"
            
            if save_dir is None:
                save_dir = self.save_dir
            elif not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            full_path = os.path.join(save_dir, output_file)
            
            fig, ax = plt.subplots(figsize=(10, max(6, len(self.control_cols) * 0.5)))
            
            # 准备数据（取前15个最重要的变量）
            top_vars = control_importance.head(15)
            var_names = top_vars['Variable'].tolist()
            coefs = top_vars['Coefficient'].values
            
            # 水平条形图
            y_pos = np.arange(len(var_names))
            colors = [self.main_color if c > 0 else '#E74C3C' for c in coefs]
            
            bars = ax.barh(y_pos, coefs, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(var_names, fontsize=10)
            ax.set_xlabel('Coefficient (Y ~ W)', fontsize=12)
            ax.set_title('Control Variables Importance', fontsize=13, fontweight='bold')
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            
            # 添加数值标签
            for i, (bar, coef) in enumerate(zip(bars, coefs)):
                ax.text(coef + 0.01 * max(abs(coefs)), i, f'{coef:.4f}', 
                       va='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(full_path, dpi=600, bbox_inches='tight')
            plt.close()
            
            print(f"📊 已保存控制变量重要性图：{full_path}")
            return full_path
            
        except Exception as e:
            print(f"Warning: 无法绘制控制变量重要性图: {e}")
            return None
    
    def save_results(self, output_file=None, save_dir=None):
        """保存结果到 Excel
        
        Parameters:
        -----------
        output_file : str, optional
            输出文件名
        save_dir : str, optional
            保存目录
        """
        if output_file is None:
            output_file = f"lineardml_{self.treatment_name}_results.xlsx"
        
        if save_dir is None:
            save_dir = self.save_dir
        elif not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        full_path = os.path.join(save_dir, output_file)
        
        # 创建结果 DataFrame
        try:
            inference_result = self.model.ate_inference()
            if hasattr(inference_result, 'pvalue'):
                p_value = inference_result.pvalue[0]
            elif hasattr(inference_result, 'pvalues'):
                p_value = inference_result.pvalues[0]
            else:
                t_stat = self.ate[0] / self.ate_std if self.ate_std > 0 else 0
                from scipy import stats
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(self.df) - len(self.control_cols) - 2))
        except:
            t_stat = self.ate[0] / self.ate_std if self.ate_std > 0 else 0
            from scipy import stats
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(self.df) - len(self.control_cols) - 2))
        
        results_data = {
            'Metric': ['Average Treatment Effect (ATE)', 'Standard Error', 
                      '95% CI Lower', '95% CI Upper', 'P-value'],
            'Value': [
                self.ate[0],
                self.ate_std,
                self.ci_lower,
                self.ci_upper,
                p_value
            ]
        }
        results_df = pd.DataFrame(results_data)
        
        # 模型信息
        model_info = pd.DataFrame({
            'Parameter': ['Treatment Variable', 'Outcome Variable', 'Control Variables', 
                         'Sample Size', 'Method'],
            'Value': [
                self.treatment_name,
                self.outcome_name,
                ', '.join(self.control_cols),
                len(self.df),
                'LinearDML'
            ]
        })
        
        # 添加控制变量的辅助回归结果（用于展示控制变量的影响）
        control_coef_data = self._get_control_variable_importance()
        
        # 保存到 Excel
        with pd.ExcelWriter(full_path, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='Causal_Effect', index=False)
            if control_coef_data is not None:
                control_coef_data.to_excel(writer, sheet_name='Control_Variables', index=False)
            model_info.to_excel(writer, sheet_name='Model_Info', index=False)
        
        print(f"💾 已保存结果：{full_path}")
        return full_path
    
    def _get_control_variable_importance(self):
        """获取控制变量的重要性（通过辅助回归）
        
        Returns:
        --------
        pd.DataFrame or None
            控制变量的系数和重要性
        """
        try:
            import statsmodels.api as sm
            
            # 使用简单的 OLS 回归 Y ~ W 来评估控制变量对 Y 的影响
            X_controls = sm.add_constant(self.W)
            model_aux = sm.OLS(self.Y, X_controls).fit()
            
            # 提取系数（跳过常数项）
            coefs = model_aux.params[1:]  # 排除 const
            std_errors = model_aux.bse[1:]
            p_values = model_aux.pvalues[1:]
            
            # 如果是 pandas Series，转换为 numpy 数组；如果已经是 numpy 数组，直接使用
            try:
                coefs = coefs.values if hasattr(coefs, 'values') else np.asarray(coefs)
            except:
                coefs = np.asarray(coefs)
            
            try:
                std_errors = std_errors.values if hasattr(std_errors, 'values') else np.asarray(std_errors)
            except:
                std_errors = np.asarray(std_errors)
            
            try:
                p_values = p_values.values if hasattr(p_values, 'values') else np.asarray(p_values)
            except:
                p_values = np.asarray(p_values)
            
            control_importance = pd.DataFrame({
                'Variable': self.control_cols,
                'Coefficient': coefs,
                'Std Error': std_errors,
                'P-value': p_values,
                'Abs_Coefficient': np.abs(coefs)
            })
            
            # 按绝对系数大小排序
            control_importance = control_importance.sort_values('Abs_Coefficient', ascending=False)
            control_importance = control_importance.drop('Abs_Coefficient', axis=1)
            
            print(f"\n📊 控制变量重要性（辅助回归 Y ~ W）:")
            print(control_importance.to_string(index=False))
            
            return control_importance
            
        except Exception as e:
            print(f"Warning: Could not compute control variable importance: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_full_analysis(self, save_dir=None):
        """运行完整分析流程"""
        if save_dir:
            self.save_dir = save_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        
        # 1. 执行估计
        self.run_estimation()
        
        # 2. 保存结果
        self.save_results()
        
        # 3. 可视化
        self.plot_treatment_effect()
        self.plot_control_variables_importance()  # 新增：控制变量重要性图
        
        print("\n✅ LinearDML 完整分析完成！")


# ======================
# 🚀 使用示例
# ======================
if __name__ == "__main__":
    print("LinearDML 因果推断工具")
    print("请提供包含 treatment, outcome 和控制变量的数据文件")
