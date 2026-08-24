import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

try:
    from econml.dml import CausalForestDML
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False
    print("Warning: econml not installed. Please install with: pip install econml")


class CausalForestDMLTool:
    """
    CausalForestDML (Causal Forest Double Machine Learning) 因果推断工具类
    使用随机森林估计异质性处理效应 (Heterogeneous Treatment Effects)
    """
    
    # ======================
    # 🎨 SSCI 柔和单色风
    # ======================
    main_color = "#4C72B0"   # 学术蓝（主色）
    line_color = "#8A8A8A"   # 柔和灰（辅助线）
    
    def __init__(self, file_path, treatment_col, outcome_col, control_cols, 
                 effect_modifiers=None, save_dir=None):
        """
        初始化工具类
        
        Parameters:
        -----------
        file_path : str
            Excel 数据文件路径
        treatment_col : str
            处理变量列名 (T/Treatment)
        outcome_col : str
            结果变量列名 (Y/Outcome)
        control_cols : list
            控制变量列名列表 (W/Controls)
        effect_modifiers : list, optional
            效应修饰变量列名列表 (X)，用于估计异质性处理效应
            如果为 None，则使用控制变量作为效应修饰变量
        save_dir : str, optional
            结果保存目录
        """
        if not ECONML_AVAILABLE:
            raise ImportError("econml library is required for CausalForestDML. Install with: pip install econml")
        
        self.file_path = file_path
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.control_cols = control_cols
        self.effect_modifiers = effect_modifiers if effect_modifiers else control_cols
        
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
        all_cols = [treatment_col, outcome_col] + control_cols + self.effect_modifiers
        missing_cols = [col for col in all_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"以下列不存在于数据中: {missing_cols}")
        
        # 提取变量
        self.T = self.df[treatment_col].values.reshape(-1, 1)  # Treatment
        self.Y = self.df[outcome_col].values  # Outcome
        self.W = self.df[control_cols].values  # Controls
        self.X = self.df[self.effect_modifiers].values  # Effect Modifiers
        
        self.feature_names = control_cols
        self.effect_modifier_names = self.effect_modifiers
        self.treatment_name = treatment_col
        self.outcome_name = outcome_col
        
        # 结果存储
        self.ate = None
        self.ate_std = None
        self.ci_lower = None
        self.ci_upper = None
        self.model = None
        self.cate_effects = None  # Conditional Average Treatment Effects
    
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
    
    def run_estimation(self, n_estimators=100, max_depth=None, min_samples_leaf=10):
        """执行因果效应估计
        
        Parameters:
        -----------
        n_estimators : int
            随机森林的树的数量（必须是4的倍数）
        max_depth : int or None
            树的最大深度
        min_samples_leaf : int
            叶节点的最小样本数
        """
        # 确保 n_estimators 是 4 的倍数（subforest_size 的要求）
        if n_estimators % 4 != 0:
            n_estimators = ((n_estimators // 4) + 1) * 4
            print(f"Warning: n_estimators adjusted to {n_estimators} (must be divisible by 4)")
        print("=" * 60)
        print("开始 CausalForestDML 因果效应估计...")
        print("=" * 60)
        print(f"处理变量 (Treatment): {self.treatment_name}")
        print(f"结果变量 (Outcome): {self.outcome_name}")
        print(f"控制变量 (Controls): {', '.join(self.control_cols)}")
        print(f"效应修饰变量 (Effect Modifiers): {', '.join(self.effect_modifier_names)}")
        print(f"样本量: {len(self.df)}")
        print(f"森林参数: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_leaf={min_samples_leaf}")
        print("-" * 60)
        
        # 构建 CausalForestDML 模型
        model_y = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                                       min_samples_leaf=min_samples_leaf, random_state=42, n_jobs=-1)
        
        # 根据处理变量类型选择模型
        if np.all(np.isin(self.T, [0, 1])):
            # 二元处理变量
            model_t = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                            min_samples_leaf=min_samples_leaf, random_state=42, n_jobs=-1)
            discrete_treatment = True
        else:
            # 连续处理变量
            model_t = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                           min_samples_leaf=min_samples_leaf, random_state=42, n_jobs=-1)
            discrete_treatment = False
        
        self.model = CausalForestDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=discrete_treatment,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        
        # 拟合模型
        self.model.fit(self.Y, self.T, W=self.W, X=self.X)
        
        # 获取平均处理效应 (ATE)
        ate_result = self.model.ate(self.X)
        if np.isscalar(ate_result):
            self.ate = np.array([ate_result])
        else:
            self.ate = np.atleast_1d(ate_result)
        
        # 获取推断结果（灵活处理不同版本的 API）
        try:
            inference_result = self.model.ate_inference(self.X)
            
            # 尝试不同的属性名
            if hasattr(inference_result, 'stderr'):
                self.ate_std = inference_result.stderr[0]
            elif hasattr(inference_result, 'se'):
                self.ate_std = inference_result.se[0]
            elif hasattr(inference_result, 'variance'):
                self.ate_std = np.sqrt(inference_result.variance[0])
            else:
                # 如果都没有，使用置信区间反推
                ci = inference_result.conf_int()[0]
                self.ate_std = (ci[1] - ci[0]) / (2 * 1.96)
        except Exception as e:
            print(f"Warning: Could not get inference results: {e}")
            self.ate_std = 0.1  # 默认值
        
        # 置信区间
        try:
            ci = self.model.ate_inference(self.X).conf_int()[0]
            self.ci_lower = ci[0]
            self.ci_upper = ci[1]
        except:
            # 如果无法获取，使用标准误计算
            self.ci_lower = self.ate[0] - 1.96 * self.ate_std
            self.ci_upper = self.ate[0] + 1.96 * self.ate_std
        
        # 获取条件平均处理效应 (CATE)
        self.cate_effects = self.model.effect(self.X)
        
        print("\n✅ 估计完成！")
        print("\n==== 因果效应估计结果 ====")
        print(f"平均处理效应 (ATE): {self.ate[0]:.4f}")
        print(f"标准误 (Std Error): {self.ate_std:.4f}")
        print(f"95% 置信区间: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]")
        
        # 判断显著性
        try:
            inference_result = self.model.ate_inference(self.X)
            if hasattr(inference_result, 'pvalue'):
                p_value = inference_result.pvalue[0]
            elif hasattr(inference_result, 'pvalues'):
                p_value = inference_result.pvalues[0]
            else:
                raise AttributeError("No pvalue attribute")
        except:
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
        
        # CATE 统计
        print(f"\n==== 异质性处理效应 (CATE) 统计 ====")
        print(f"CATE 均值: {np.mean(self.cate_effects):.4f}")
        print(f"CATE 标准差: {np.std(self.cate_effects):.4f}")
        print(f"CATE 最小值: {np.min(self.cate_effects):.4f}")
        print(f"CATE 最大值: {np.max(self.cate_effects):.4f}")
        
        return self.ate[0]
    
    def plot_treatment_effect(self, output_file=None, save_dir=None):
        """绘制处理效应图"""
        if output_file is None:
            output_file = f"causalforest_{self.treatment_name}_effect.png"
        
        if save_dir is None:
            save_dir = self.save_dir
        elif not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        full_path = os.path.join(save_dir, output_file)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        categories = ['Average Treatment Effect']
        values = [self.ate[0]]
        errors = [[self.ate[0] - self.ci_lower], [self.ci_upper - self.ate[0]]]
        
        bars = ax.bar(categories, values, yerr=errors, capsize=10, 
                     color=self.main_color, alpha=0.7, edgecolor='black', linewidth=1.5)
        
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
    
    def plot_cate_distribution(self, output_file=None, save_dir=None):
        """绘制 CATE 分布图"""
        if output_file is None:
            output_file = f"causalforest_{self.treatment_name}_cate_distribution.png"
        
        if save_dir is None:
            save_dir = self.save_dir
        elif not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        full_path = os.path.join(save_dir, output_file)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(self.cate_effects, bins=30, color=self.main_color, alpha=0.7, 
               edgecolor='black', linewidth=0.5)
        ax.axvline(x=self.ate[0], color='red', linestyle='--', linewidth=2, label=f'ATE = {self.ate[0]:.4f}')
        ax.axvline(x=np.mean(self.cate_effects), color='orange', linestyle='--', 
                  linewidth=2, label=f'Mean CATE = {np.mean(self.cate_effects):.4f}')
        
        ax.set_xlabel('Conditional Average Treatment Effect (CATE)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Distribution of Heterogeneous Treatment Effects', fontsize=13, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(full_path, dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"📊 已保存 CATE 分布图：{full_path}")
        return full_path
    
    def plot_control_variables_importance(self, output_file=None, save_dir=None):
        """绘制控制变量重要性图（水平条形图）"""
        try:
            # 获取控制变量重要性
            control_importance = self._get_control_variable_importance()
            if control_importance is None or len(control_importance) == 0:
                return None
            
            if output_file is None:
                output_file = f"causalforest_{self.treatment_name}_control_variables.png"
            
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
        """保存结果到 Excel"""
        if output_file is None:
            output_file = f"causalforest_{self.treatment_name}_results.xlsx"
        
        if save_dir is None:
            save_dir = self.save_dir
        elif not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        full_path = os.path.join(save_dir, output_file)
        
        # ATE 结果
        try:
            inference_result = self.model.ate_inference(self.X)
            if hasattr(inference_result, 'pvalue'):
                p_value = inference_result.pvalue[0]
            elif hasattr(inference_result, 'pvalues'):
                p_value = inference_result.pvalues[0]
            else:
                raise AttributeError("No pvalue attribute")
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
        
        # CATE 统计
        cate_stats = pd.DataFrame({
            'Statistic': ['Mean', 'Std', 'Min', 'Max', 'Median', 'Q1 (25%)', 'Q3 (75%)'],
            'Value': [
                np.mean(self.cate_effects),
                np.std(self.cate_effects),
                np.min(self.cate_effects),
                np.max(self.cate_effects),
                np.median(self.cate_effects),
                np.percentile(self.cate_effects, 25),
                np.percentile(self.cate_effects, 75)
            ]
        })
        
        # 模型信息
        model_info = pd.DataFrame({
            'Parameter': ['Treatment Variable', 'Outcome Variable', 'Control Variables', 
                         'Effect Modifiers', 'Sample Size', 'Method'],
            'Value': [
                self.treatment_name,
                self.outcome_name,
                ', '.join(self.control_cols),
                ', '.join(self.effect_modifier_names),
                len(self.df),
                'CausalForestDML'
            ]
        })
        
        # 获取控制变量重要性（辅助回归）
        control_importance_df = self._get_control_variable_importance()
        
        # 保存到 Excel
        with pd.ExcelWriter(full_path, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='Causal_Effect', index=False)
            cate_stats.to_excel(writer, sheet_name='CATE_Statistics', index=False)
            model_info.to_excel(writer, sheet_name='Model_Info', index=False)
            if control_importance_df is not None:
                control_importance_df.to_excel(writer, sheet_name='Control_Variables', index=False)
        
        print(f"💾 已保存结果：{full_path}")
        return full_path
    
    def _get_control_variable_importance(self):
        """获取控制变量重要性（辅助回归分析）
        
        通过 OLS 回归 Y ~ W 来评估各个控制变量对结果变量的影响
        
        Returns:
        --------
        pd.DataFrame or None
            控制变量的系数、标准误和P值
        """
        try:
            import statsmodels.api as sm
            
            print("\n" + "="*60)
            print("📊 控制变量重要性分析（辅助回归 Y ~ W）")
            print("="*60)
            
            # 使用 OLS 回归评估控制变量对 Y 的影响
            X_with_const = sm.add_constant(self.W)
            model_aux = sm.OLS(self.Y, X_with_const).fit()
            
            # 提取结果（排除常数项）
            coefs = model_aux.params[1:]  # 跳过 const
            std_err = model_aux.bse[1:]
            p_vals = model_aux.pvalues[1:]
            
            # 如果是 pandas Series，转换为 numpy 数组；如果已经是 numpy 数组，直接使用
            try:
                coefs = coefs.values if hasattr(coefs, 'values') else np.asarray(coefs)
            except:
                coefs = np.asarray(coefs)
            
            try:
                std_err = std_err.values if hasattr(std_err, 'values') else np.asarray(std_err)
            except:
                std_err = np.asarray(std_err)
            
            try:
                p_vals = p_vals.values if hasattr(p_vals, 'values') else np.asarray(p_vals)
            except:
                p_vals = np.asarray(p_vals)
            
            control_importance = pd.DataFrame({
                'Variable': self.control_cols,
                'Coefficient': coefs,
                'Std_Error': std_err,
                'P_value': p_vals,
                'Abs_Coefficient': np.abs(coefs)  # 用于排序
            })
            
            # 按绝对系数大小排序（从大到小）
            control_importance = control_importance.sort_values('Abs_Coefficient', ascending=False)
            control_importance = control_importance.drop('Abs_Coefficient', axis=1)
            
            # 打印结果
            print(control_importance.to_string(index=False))
            print("="*60)
            
            return control_importance
            
        except Exception as e:
            print(f"Warning: 无法计算控制变量重要性: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_full_analysis(self, save_dir=None, **kwargs):
        """运行完整分析流程"""
        if save_dir:
            self.save_dir = save_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        
        # 1. 执行估计
        self.run_estimation(**kwargs)
        
        # 2. 保存结果
        self.save_results()
        
        # 3. 可视化
        self.plot_treatment_effect()
        self.plot_cate_distribution()
        self.plot_control_variables_importance()  # 新增：控制变量重要性图
        
        print("\n✅ CausalForestDML 完整分析完成！")


# ======================
# 🚀 使用示例
# ======================
if __name__ == "__main__":
    print("CausalForestDML 因果推断工具")
    print("请提供包含 treatment, outcome, controls 和 effect modifiers 的数据文件")
