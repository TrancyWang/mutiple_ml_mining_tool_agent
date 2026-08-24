import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os


class AdaBoostRegressionTool:
    """
    AdaBoost 回归分析工具类
    支持多次交叉验证、SHAP 解释性分析和可视化
    """
    
    # ======================
    # 🎨 SSCI 柔和单色风（非黑灰）
    # ======================
    main_color = "#4C72B0"   # 学术蓝（主色）
    line_color = "#8A8A8A"   # 柔和灰（辅助线）
    
    def __init__(self, file_path, multiple_folder=5, save_dir=None, shap_save_dir=None):
        """
        初始化工具类
        
        Parameters:
        -----------
        file_path : str
            Excel 数据文件路径
        multiple_folder : int
            交叉验证次数，默认 5 次
        save_dir : str, optional
            图片保存目录，默认为 text_mining_tools_agent_client_cpu/data_analysis_result/imgs
        shap_save_dir : str, optional
            SHAP 特征重要性 Excel 文件保存目录，默认为 text_mining_tools_agent_client_cpu/data_analysis_result/shap_features
        """
        self.file_path = file_path
        self.multiple_folder = multiple_folder
        
        # 设置默认图片保存目录
        if save_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            self.save_dir = os.path.join(project_root, 'data_analysis_result/imgs')
        else:
            self.save_dir = save_dir
        
        # 设置默认 SHAP 文件保存目录
        if shap_save_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            self.shap_save_dir = os.path.join(project_root, 'data_analysis_result', 'shap_features')
        else:
            self.shap_save_dir = shap_save_dir
        
        # 创建保存目录（如果不存在）
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.shap_save_dir):
            os.makedirs(self.shap_save_dir)
        
        # 设置绘图样式
        self._setup_plot_style()
        
        # 读取数据
        self.df = pd.read_excel(file_path)
        self.X = self.df.iloc[:, :-1]
        self.y = self.df.iloc[:, -1]
        self.feature_names = self.X.columns.tolist()
        
        # 结果存储
        self.r2_list, self.rmse_list, self.mae_list = [], [], []
        self.shap_list = []
        self.model = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
    
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
    
    def run_multiple_training(self):
        """执行多次交叉验证训练"""
        print(f"开始 {self.multiple_folder} 次交叉验证训练...")
        
        for i in range(self.multiple_folder):
            # 训练集测试集划分
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42 + i
            )
            
            # 构建 AdaBoost 回归模型
            base_estimator = DecisionTreeRegressor(max_depth=3, random_state=42 + i)
            model = AdaBoostRegressor(
                estimator=base_estimator,
                n_estimators=200,
                learning_rate=0.1,
                random_state=42 + i
            )
            
            # 模型训练
            model.fit(X_train, y_train)
            
            # 模型评估
            y_pred = model.predict(X_test)
            
            # 指标计算
            self.r2_list.append(r2_score(y_test, y_pred))
            self.rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            self.mae_list.append(mean_absolute_error(y_test, y_pred))
            
            # SHAP 计算 - AdaBoost 需要使用 KernelExplainer 或基于特征重要性
            # 由于 AdaBoostRegressor 不直接被 TreeExplainer 支持，我们使用模型的 feature_importances_
            try:
                # 尝试使用 TreeExplainer（某些版本可能支持）
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
                self.shap_list.append(np.abs(shap_values).mean(axis=0))
            except Exception as e:
                print(f"Warning: TreeExplainer not supported for AdaBoost, using feature importances instead: {e}")
                # 使用模型的特征重要性作为替代
                feature_importance = model.feature_importances_
                self.shap_list.append(feature_importance)
            
            # 保存最后一次的结果用于可视化
            if i == self.multiple_folder - 1:
                self.model = model
                self.X_test = X_test
                self.y_test = y_test
                self.y_pred = y_pred
        
        # 平均结果
        self.r2_mean = np.mean(self.r2_list)
        self.rmse_mean = np.mean(self.rmse_list)
        self.mae_mean = np.mean(self.mae_list)
        self.shap_mean = np.mean(self.shap_list, axis=0)
        
        print("==== Final Averaged Results ====")
        print(f"R²     = {self.r2_mean:.4f}")
        print(f"RMSE   = {self.rmse_mean:.4f}")
        print(f"MAE    = {self.mae_mean:.4f}")
    
    def save_shap_importance(self, output_file=None, save_dir=None):
        """保存 SHAP 重要性和模型指标到 Excel（多 Sheet）
        
        Parameters:
        -----------
        output_file : str, optional
            输出文件名，默认为 adaboost_shap_importance_mean.xlsx
        save_dir : str, optional
            SHAP 文件保存目录，默认为实例初始化时设置的 shap_save_dir
        """
        if output_file is None:
            output_file = "adaboost_shap_importance_mean.xlsx"
        
        # 确定保存目录
        if save_dir is None:
            save_dir = self.shap_save_dir
        elif not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 完整文件路径
        full_path = os.path.join(save_dir, output_file)
        
        # Sheet 1: SHAP 重要性
        shap_importance = pd.DataFrame({
            "Feature": self.feature_names,
            "MeanAbsSHAP": self.shap_mean
        }).sort_values(by="MeanAbsSHAP", ascending=False)
        
        # Sheet 2: 模型评估指标
        metrics_data = pd.DataFrame({
            'Metric': ['R²', 'RMSE', 'MAE'],
            'Value': [self.r2_mean, self.rmse_mean, self.mae_mean]
        })
        
        # 使用 ExcelWriter 写入多个 sheet
        with pd.ExcelWriter(full_path, engine='openpyxl') as writer:
            shap_importance.to_excel(writer, sheet_name='SHAP_Importance', index=False)
            metrics_data.to_excel(writer, sheet_name='Model_Metrics', index=False)
        
        print("\nSHAP Importance (Mean):")
        print(shap_importance)
        
        print("\nModel Metrics:")
        print(metrics_data)
        
        print(f"\n已保存 SHAP 文件：{full_path}")
        
        return shap_importance
    
    def plot_actual_vs_predicted(self, output_file=None, save_dir=None):
        """绘制实际值 vs 预测值图
        
        Parameters:
        -----------
        output_file : str, optional
            输出文件名，默认为 adaboost_figure_actual_vs_predicted.png
        save_dir : str, optional
            图片保存目录，默认为实例初始化时设置的 save_dir
        """
        if output_file is None:
            output_file = "adaboost_figure_actual_vs_predicted.png"
        
        # 确定保存目录
        if save_dir is None:
            save_dir = self.save_dir
        elif not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 完整文件路径
        full_path = os.path.join(save_dir, output_file)
        
        plt.figure(figsize=(6, 6))
        
        plt.scatter(
            self.y_test, self.y_pred,
            color=self.main_color,
            alpha=0.6,
            s=35
        )
        
        # 理想线
        plt.plot(
            [self.y_test.min(), self.y_test.max()],
            [self.y_test.min(), self.y_test.max()],
            linestyle='--',
            color=self.line_color,
            linewidth=1
        )
        
        # 回归线
        z = np.polyfit(self.y_test, self.y_pred, 1)
        p = np.poly1d(z)
        plt.plot(self.y_test, p(self.y_test), color=self.main_color, linewidth=1.5)
        
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        
        # 指标标注
        plt.text(
            0.05, 0.95,
            f"$R^2$={self.r2_mean:.3f}\nRMSE={self.rmse_mean:.3f}",
            transform=plt.gca().transAxes,
            va='top'
        )
        
        sns.despine()
        plt.tight_layout()
        plt.savefig(full_path, dpi=600)
        plt.close()
        
        print(f"已保存：{full_path}")
    
    def plot_residual_analysis(self, output_file=None, save_dir=None):
        """绘制残差分析图
        
        Parameters:
        -----------
        output_file : str, optional
            输出文件名，默认为 adaboost_figure_residual_plot.png
        save_dir : str, optional
            图片保存目录，默认为实例初始化时设置的 save_dir
        """
        if output_file is None:
            output_file = "adaboost_figure_residual_plot.png"
        
        # 确定保存目录
        if save_dir is None:
            save_dir = self.save_dir
        elif not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 完整文件路径
        full_path = os.path.join(save_dir, output_file)
        
        residuals = self.y_test - self.y_pred
        
        plt.figure(figsize=(6, 6))
        
        plt.scatter(
            self.y_pred, residuals,
            color=self.main_color,
            alpha=0.6,
            s=35
        )
        
        # 零线
        plt.axhline(0, linestyle='--', color=self.line_color, linewidth=1)
        
        # 趋势线
        z = np.polyfit(self.y_pred, residuals, 1)
        p = np.poly1d(z)
        plt.plot(self.y_pred, p(self.y_pred), color=self.main_color, linewidth=1.5)
        
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        
        sns.despine()
        plt.tight_layout()
        plt.savefig(full_path, dpi=600)
        plt.close()
        
        print(f"已保存：{full_path}")
    
    def plot_shap_summary(self, output_file=None, max_display=12, save_dir=None):
        """绘制 SHAP Summary 图
        
        Parameters:
        -----------
        output_file : str, optional
            输出文件名，默认为 adaboost_figure_shap_summary.png
        max_display : int, optional
            显示的特征数量，默认 12
        save_dir : str, optional
            图片保存目录，默认为实例初始化时设置的 save_dir
        """
        if output_file is None:
            output_file = "adaboost_figure_shap_summary.png"
        
        # 确定保存目录
        if save_dir is None:
            save_dir = self.save_dir
        elif not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 完整文件路径
        full_path = os.path.join(save_dir, output_file)
        
        plt.figure()
        
        # AdaBoost 的 SHAP Summary 图 - 使用特征重要性代替
        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(self.X_test)
            
            shap.summary_plot(
                shap_values,
                self.X_test,
                max_display=max_display,
                show=False
            )
        except Exception as e:
            print(f"Warning: Cannot generate SHAP summary plot for AdaBoost, using feature importance bar plot instead: {e}")
            # 使用特征重要性绘制条形图
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[-max_display:]  # 取最重要的 max_display 个特征
            
            plt.barh(range(len(indices)), importances[indices], color=self.main_color)
            plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices])
            plt.xlabel('Feature Importance')
            plt.title('Feature Importances (AdaBoost)')
            plt.tight_layout()
        
        plt.savefig(full_path, dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"已保存：{full_path}")
    
    def run_full_analysis(self, save_dir=None):
        """
        运行完整分析流程
        
        Parameters:
        -----------
        save_dir : str, optional
            结果保存目录，默认为当前目录
        """
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            old_dir = os.getcwd()
            os.chdir(save_dir)
        else:
            old_dir = None
        
        try:
            # 1. 多次训练
            self.run_multiple_training()
            
            # 2. 保存 SHAP 重要性
            self.save_shap_importance()
            
            # 3. 可视化
            self.plot_actual_vs_predicted()
            self.plot_residual_analysis()
            self.plot_shap_summary()
            
            print("\n✅ 完整分析完成！")
            
        finally:
            if old_dir:
                os.chdir(old_dir)


# ======================
# 🚀 使用示例
# ======================
if __name__ == "__main__":
    # 创建工具实例
    tool = AdaBoostRegressionTool(
        file_path="ml_test_data.xlsx",
        multiple_folder=5
    )
    
    # 运行完整分析
    tool.run_full_analysis()
