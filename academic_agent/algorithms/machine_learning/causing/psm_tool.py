import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')


class PSMTool:
    """
    PSM (Propensity Score Matching) 倾向得分匹配工具类
    用于估计处理变量的平均处理效应 (ATT)
    """
    
    main_color = "#4C72B0"
    line_color = "#8A8A8A"
    
    def __init__(self, file_path, treatment_col, outcome_col, control_cols, 
                 matching_ratio=1, caliper=None, save_dir=None):
        """
        Parameters:
        -----------
        file_path : str
            数据文件路径
        treatment_col : str
            处理变量列名 (应该是二元变量: 0或1)
        outcome_col : str
            结果变量列名
        control_cols : list
            控制变量列名列表（用于计算倾向得分）
        matching_ratio : int
            每个处理组样本匹配的对照组样本数
        caliper : float, optional
            卡尺宽度（倾向得分的标准差倍数），通常为0.2
        save_dir : str, optional
            保存目录
        """
        self.file_path = file_path
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.control_cols = control_cols
        self.matching_ratio = matching_ratio
        self.caliper = caliper
        
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
        self.df = pd.read_excel(file_path).copy()
        
        # 验证列
        all_cols = [treatment_col, outcome_col] + control_cols
        missing_cols = [col for col in all_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"以下列不存在于数据中: {missing_cols}")
        
        # 验证处理变量是二元的
        if not np.all(np.isin(self.df[treatment_col].unique(), [0, 1])):
            raise ValueError("处理变量必须是二元变量 (0 或 1)")
        
        self.T = self.df[treatment_col].values
        self.Y = self.df[outcome_col].values
        self.W = self.df[control_cols].values
        
        self.feature_names = control_cols
        self.treatment_name = treatment_col
        self.outcome_name = outcome_col
        
        # 结果存储
        self.propensity_scores = None
        self.matched_data = None
        self.att = None  # Average Treatment Effect on the Treated
        self.se_att = None
        self.balance_before = None
        self.balance_after = None
    
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
    
    def estimate_propensity_scores(self):
        """估计倾向得分"""
        print("步骤 1/4: 估计倾向得分...")
        
        # 使用逻辑回归估计倾向得分
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(self.W, self.T)
        
        self.propensity_scores = lr_model.predict_proba(self.W)[:, 1]
        self.df['propensity_score'] = self.propensity_scores
        
        print(f"✓ 倾向得分估计完成")
        print(f"  处理组样本数: {np.sum(self.T == 1)}")
        print(f"  对照组样本数: {np.sum(self.T == 0)}")
        print(f"  倾向得分范围: [{self.propensity_scores.min():.4f}, {self.propensity_scores.max():.4f}]")
        
        return self.propensity_scores
    
    def perform_matching(self):
        """执行倾向得分匹配"""
        print("\n步骤 2/4: 执行倾向得分匹配...")
        
        # 分离处理组和对照组
        treated_idx = np.where(self.T == 1)[0]
        control_idx = np.where(self.T == 0)[0]
        
        treated_scores = self.propensity_scores[treated_idx].reshape(-1, 1)
        control_scores = self.propensity_scores[control_idx].reshape(-1, 1)
        
        # 如果设置了卡尺
        if self.caliper is not None:
            caliper_value = self.caliper * self.propensity_scores.std()
            print(f"  卡尺宽度: {caliper_value:.4f} ({self.caliper} × SD)")
        
        # 使用最近邻匹配
        nbrs = NearestNeighbors(n_neighbors=self.matching_ratio, metric='euclidean')
        nbrs.fit(control_scores)
        
        distances, indices = nbrs.kneighbors(treated_scores)
        
        # 应用卡尺
        if self.caliper is not None:
            caliper_value = self.caliper * self.propensity_scores.std()
            mask = distances[:, 0] <= caliper_value
            treated_idx = treated_idx[mask]
            indices = indices[mask]
            distances = distances[mask]
            print(f"  卡尺内匹配成功的样本数: {np.sum(mask)}/{len(mask)}")
        
        # 构建匹配后的数据集
        matched_control_idx = control_idx[indices.flatten()]
        
        treated_data = self.df.iloc[treated_idx].copy()
        treated_data['match_type'] = 'treated'
        
        control_data = self.df.iloc[matched_control_idx].copy()
        control_data['match_type'] = 'matched_control'
        
        self.matched_data = pd.concat([treated_data, control_data], ignore_index=True)
        
        print(f"✓ 匹配完成")
        print(f"  匹配后处理组样本数: {len(treated_data)}")
        print(f"  匹配后对照组样本数: {len(control_data)}")
        
        return self.matched_data
    
    def check_balance(self):
        """检查协变量平衡"""
        print("\n步骤 3/4: 检查协变量平衡...")
        
        # 匹配前的平衡
        balance_before = {}
        for col in self.control_cols:
            treated_mean = self.df[self.T == 1][col].mean()
            control_mean = self.df[self.T == 0][col].mean()
            treated_std = self.df[self.T == 1][col].std()
            control_std = self.df[self.T == 0][col].std()
            
            # 标准化均值差
            pooled_std = np.sqrt((treated_std**2 + control_std**2) / 2)
            if pooled_std > 0:
                smd = (treated_mean - control_mean) / pooled_std
            else:
                smd = 0
            
            balance_before[col] = smd
        
        # 匹配后的平衡
        balance_after = {}
        for col in self.control_cols:
            treated_mean = self.matched_data[self.matched_data['match_type'] == 'treated'][col].mean()
            control_mean = self.matched_data[self.matched_data['match_type'] == 'matched_control'][col].mean()
            treated_std = self.matched_data[self.matched_data['match_type'] == 'treated'][col].std()
            control_std = self.matched_data[self.matched_data['match_type'] == 'matched_control'][col].std()
            
            pooled_std = np.sqrt((treated_std**2 + control_std**2) / 2)
            if pooled_std > 0:
                smd = (treated_mean - control_mean) / pooled_std
            else:
                smd = 0
            
            balance_after[col] = smd
        
        self.balance_before = balance_before
        self.balance_after = balance_after
        
        print("✓ 平衡性检验完成")
        print("\n标准化均值差 (SMD):")
        print(f"{'变量':<20} {'匹配前':>10} {'匹配后':>10} {'改善':>10}")
        print("-" * 55)
        for col in self.control_cols:
            before = self.balance_before[col]
            after = self.balance_after[col]
            improvement = abs(before) - abs(after)
            print(f"{col:<20} {before:>10.4f} {after:>10.4f} {improvement:>10.4f}")
        
        return balance_before, balance_after
    
    def estimate_att(self):
        """估计处理组的平均处理效应 (ATT)"""
        print("\n步骤 4/4: 估计 ATT...")
        
        treated_outcomes = self.matched_data[self.matched_data['match_type'] == 'treated'][self.outcome_col]
        control_outcomes = self.matched_data[self.matched_data['match_type'] == 'matched_control'][self.outcome_col]
        
        # ATT = E[Y(1) - Y(0) | T=1]
        self.att = treated_outcomes.mean() - control_outcomes.mean()
        
        # 标准误
        n_treated = len(treated_outcomes)
        n_control = len(control_outcomes)
        
        var_treated = treated_outcomes.var() / n_treated
        var_control = control_outcomes.var() / n_control
        self.se_att = np.sqrt(var_treated + var_control)
        
        # 置信区间
        ci_lower = self.att - 1.96 * self.se_att
        ci_upper = self.att + 1.96 * self.se_att
        
        # P值
        t_stat = self.att / self.se_att
        from scipy import stats
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_treated + n_control - 2))
        
        print(f"✓ ATT 估计完成")
        print(f"\n==== 因果效应估计结果 ====")
        print(f"平均处理效应 (ATT): {self.att:.4f}")
        print(f"标准误 (Std Error): {self.se_att:.4f}")
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
        
        return self.att
    
    def plot_balance(self, output_file=None, save_dir=None):
        """绘制平衡图"""
        if output_file is None:
            output_file = f"psm_{self.treatment_name}_balance.png"
        
        if save_dir is None:
            save_dir = self.save_dir
        elif not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        full_path = os.path.join(save_dir, output_file)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        variables = list(self.balance_before.keys())
        before_values = [self.balance_before[var] for var in variables]
        after_values = [self.balance_after[var] for var in variables]
        
        x = np.arange(len(variables))
        width = 0.35
        
        bars1 = ax.barh(x - width/2, before_values, width, label='Before Matching', 
                       color='#E74C3C', alpha=0.7)
        bars2 = ax.barh(x + width/2, after_values, width, label='After Matching', 
                       color=self.main_color, alpha=0.7)
        
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.axvline(x=0.1, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axvline(x=-0.1, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        
        ax.set_yticks(x)
        ax.set_yticklabels(variables, fontsize=10)
        ax.set_xlabel('Standardized Mean Difference', fontsize=12)
        ax.set_title('Covariate Balance Before and After Matching', fontsize=13, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(full_path, dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 已保存平衡图：{full_path}")
        return full_path
    
    def plot_ps_distribution(self, output_file=None, save_dir=None):
        """绘制倾向得分分布图"""
        if output_file is None:
            output_file = f"psm_{self.treatment_name}_ps_distribution.png"
        
        if save_dir is None:
            save_dir = self.save_dir
        elif not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        full_path = os.path.join(save_dir, output_file)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        treated_ps = self.df[self.T == 1]['propensity_score']
        control_ps = self.df[self.T == 0]['propensity_score']
        
        ax.hist(treated_ps, bins=30, alpha=0.5, label='Treated', color=self.main_color, density=True)
        ax.hist(control_ps, bins=30, alpha=0.5, label='Control', color='#E74C3C', density=True)
        
        ax.set_xlabel('Propensity Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Propensity Score Distribution', fontsize=13, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(full_path, dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"📊 已保存倾向得分分布图：{full_path}")
        return full_path
    
    def save_results(self, output_file=None, save_dir=None):
        """保存结果到 Excel"""
        if output_file is None:
            output_file = f"psm_{self.treatment_name}_results.xlsx"
        
        if save_dir is None:
            save_dir = self.save_dir
        elif not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        full_path = os.path.join(save_dir, output_file)
        
        # ATT 结果
        att_data = {
            'Metric': ['Average Treatment Effect on Treated (ATT)', 'Standard Error', 
                      '95% CI Lower', '95% CI Upper'],
            'Value': [
                self.att,
                self.se_att,
                self.att - 1.96 * self.se_att,
                self.att + 1.96 * self.se_att
            ]
        }
        att_df = pd.DataFrame(att_data)
        
        # 平衡性检验
        balance_data = {
            'Variable': list(self.balance_before.keys()),
            'SMD_Before': list(self.balance_before.values()),
            'SMD_After': list(self.balance_after.values()),
            'Improvement': [abs(self.balance_before[v]) - abs(self.balance_after[v]) 
                          for v in self.balance_before.keys()]
        }
        balance_df = pd.DataFrame(balance_data)
        
        # 模型信息
        model_info = pd.DataFrame({
            'Parameter': ['Treatment Variable', 'Outcome Variable', 'Control Variables', 
                         'Matching Ratio', 'Caliper', 'Sample Size (Original)',
                         'Sample Size (Matched)', 'Method'],
            'Value': [
                self.treatment_name,
                self.outcome_name,
                ', '.join(self.control_cols),
                self.matching_ratio,
                self.caliper if self.caliper else 'None',
                len(self.df),
                len(self.matched_data),
                'Propensity Score Matching'
            ]
        })
        
        with pd.ExcelWriter(full_path, engine='openpyxl') as writer:
            att_df.to_excel(writer, sheet_name='ATT_Estimate', index=False)
            balance_df.to_excel(writer, sheet_name='Balance_Check', index=False)
            model_info.to_excel(writer, sheet_name='Model_Info', index=False)
            self.matched_data.to_excel(writer, sheet_name='Matched_Data', index=False)
        
        print(f"💾 已保存结果：{full_path}")
        return full_path
    
    def run_full_analysis(self, save_dir=None):
        """运行完整分析流程"""
        if save_dir:
            self.save_dir = save_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        
        # 1. 估计倾向得分
        self.estimate_propensity_scores()
        
        # 2. 执行匹配
        self.perform_matching()
        
        # 3. 检查平衡
        self.check_balance()
        
        # 4. 估计 ATT
        self.estimate_att()
        
        # 5. 保存结果
        self.save_results()
        
        # 6. 可视化
        self.plot_balance()
        self.plot_ps_distribution()
        
        print("\n✅ PSM 完整分析完成！")


if __name__ == "__main__":
    print("PSM (倾向得分匹配) 因果推断工具")
