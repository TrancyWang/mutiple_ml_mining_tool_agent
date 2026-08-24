import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os


class SVMClassificationTool:
    """
    SVM 支持向量机分类分析工具类
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
        self.class_names = self.y.unique().tolist()
        
        # 结果存储
        self.accuracy_list, self.precision_list, self.recall_list, self.f1_list = [], [], [], []
        self.shap_list = []
        self.model = None
        self.scaler = None
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
            
            # 特征标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 构建 SVM 分类模型
            model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,  # 启用概率估计以支持 SHAP
                random_state=42 + i,
                class_weight='balanced'  # 处理类别不平衡
            )
            
            # 模型训练
            model.fit(X_train_scaled, y_train)
            
            # 模型预测
            y_pred = model.predict(X_test_scaled)
            
            # 指标计算
            self.accuracy_list.append(accuracy_score(y_test, y_pred))
            self.precision_list.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            self.recall_list.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            self.f1_list.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            
            # SHAP 计算（使用 KernelExplainer）
            if i == self.multiple_folder - 1:  # 只在最后一次计算 SHAP
                explainer = shap.KernelExplainer(model.predict_proba, X_train_scaled[:100])  # 使用部分样本加速
                shap_values = explainer.shap_values(X_test_scaled[:50])  # 使用部分样本加速
                
                # 对于多分类，取平均绝对 SHAP 值
                if isinstance(shap_values, list):
                    shap_mean = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
                else:
                    shap_array = np.asarray(shap_values)
                    # SHAP 0.46+ returns (samples, features, classes) for
                    # predict_proba. Collapse both samples and classes so the
                    # result remains one importance value per feature, as the
                    # workbook and plotting methods require.
                    if shap_array.ndim == 3:
                        shap_mean = np.abs(shap_array).mean(axis=(0, 2))
                    else:
                        shap_mean = np.abs(shap_array).mean(axis=0)
                
                self.shap_list.append(shap_mean)
            
            # 保存最后一次的结果用于可视化
            if i == self.multiple_folder - 1:
                self.model = model
                self.scaler = scaler
                self.X_test = X_test_scaled
                self.y_test = y_test
                self.y_pred = y_pred
        
        # 平均结果
        self.accuracy_mean = np.mean(self.accuracy_list)
        self.precision_mean = np.mean(self.precision_list)
        self.recall_mean = np.mean(self.recall_list)
        self.f1_mean = np.mean(self.f1_list)
        self.shap_mean = np.mean(self.shap_list, axis=0) if self.shap_list else None
        
        print("==== Final Averaged Results ====")
        print(f"Accuracy  = {self.accuracy_mean:.4f}")
        print(f"Precision = {self.precision_mean:.4f}")
        print(f"Recall    = {self.recall_mean:.4f}")
        print(f"F1-Score  = {self.f1_mean:.4f}")
    
    def save_shap_importance(self, output_file=None, save_dir=None):
        """保存 SHAP 重要性和模型指标到 Excel（多 Sheet）
        
        Parameters:
        -----------
        output_file : str, optional
            输出文件名，默认为 svm_shap_importance_mean.xlsx
        save_dir : str, optional
            SHAP 文件保存目录，默认为实例初始化时设置的 shap_save_dir
        """
        if output_file is None:
            output_file = "svm_shap_importance_mean.xlsx"
        
        # 确定保存目录
        if save_dir is None:
            save_dir = self.shap_save_dir
        elif not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 完整文件路径
        full_path = os.path.join(save_dir, output_file)
        
        # Sheet 1: SHAP 重要性
        if self.shap_mean is not None:
            shap_importance = pd.DataFrame({
                "Feature": self.feature_names,
                "MeanAbsSHAP": self.shap_mean
            }).sort_values(by="MeanAbsSHAP", ascending=False)
        else:
            shap_importance = pd.DataFrame(columns=["Feature", "MeanAbsSHAP"])
        
        # Sheet 2: 模型评估指标
        metrics_data = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Value': [self.accuracy_mean, self.precision_mean, self.recall_mean, self.f1_mean]
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
    
    def plot_confusion_matrix(self, output_file=None, save_dir=None):
        """绘制混淆矩阵图
        
        Parameters:
        -----------
        output_file : str, optional
            输出文件名，默认为 svm_figure_confusion_matrix.png
        save_dir : str, optional
            图片保存目录，默认为实例初始化时设置的 save_dir
        """
        if output_file is None:
            output_file = "svm_figure_confusion_matrix.png"
        
        # 确定保存目录
        if save_dir is None:
            save_dir = self.save_dir
        elif not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 完整文件路径
        full_path = os.path.join(save_dir, output_file)
        
        # 计算混淆矩阵
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        
        # 绘制热力图
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            linewidths=0.5,
            linecolor='gray'
        )
        
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, pad=15)
        
        plt.tight_layout()
        plt.savefig(full_path, dpi=600)
        plt.close()
        
        print(f"已保存：{full_path}")
    
    def plot_classification_report(self, output_file=None, save_dir=None):
        """绘制分类报告柱状图
        
        Parameters:
        -----------
        output_file : str, optional
            输出文件名，默认为 svm_figure_classification_report.png
        save_dir : str, optional
            图片保存目录，默认为实例初始化时设置的 save_dir
        """
        if output_file is None:
            output_file = "svm_figure_classification_report.png"
        
        # 确定保存目录
        if save_dir is None:
            save_dir = self.save_dir
        elif not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 完整文件路径
        full_path = os.path.join(save_dir, output_file)
        
        # 获取每个类别的详细指标
        report = classification_report(self.y_test, self.y_pred, output_dict=True)
        
        # 提取每个类别的 precision, recall, f1-score
        classes = [cls for cls in report.keys() if cls not in ['accuracy', 'macro avg', 'weighted avg']]
        metrics = ['precision', 'recall', 'f1-score']
        
        data = []
        for cls in classes:
            for metric in metrics:
                data.append({
                    'Class': str(cls),
                    'Metric': metric,
                    'Value': report[cls][metric]
                })
        
        df_plot = pd.DataFrame(data)
        
        plt.figure(figsize=(10, 6))
        
        # 绘制分组柱状图
        sns.barplot(
            data=df_plot,
            x='Class',
            y='Value',
            hue='Metric',
            palette=[self.main_color, '#DD5145', '#5CB85C']
        )
        
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Classification Report by Class', fontsize=14, pad=15)
        plt.legend(title='Metric', loc='lower right')
        plt.ylim(0, 1.1)
        
        sns.despine()
        plt.tight_layout()
        plt.savefig(full_path, dpi=600)
        plt.close()
        
        print(f"已保存：{full_path}")
    
    def plot_feature_importance(self, output_file=None, max_display=12, save_dir=None):
        """绘制特征重要性柱状图（基于 SHAP）
        
        Parameters:
        -----------
        output_file : str, optional
            输出文件名，默认为 svm_figure_feature_importance.png
        max_display : int, optional
            显示的特征数量，默认 12
        save_dir : str, optional
            图片保存目录，默认为实例初始化时设置的 save_dir
        """
        if output_file is None:
            output_file = "svm_figure_feature_importance.png"
        
        # 确定保存目录
        if save_dir is None:
            save_dir = self.save_dir
        elif not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 完整文件路径
        full_path = os.path.join(save_dir, output_file)
        
        if self.shap_mean is None:
            print("警告：SHAP 值未计算，跳过特征重要性图")
            return
        
        # 创建特征重要性 DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.shap_mean
        }).sort_values(by='Importance', ascending=True).tail(max_display)
        
        plt.figure(figsize=(10, 8))
        
        # 绘制水平柱状图
        plt.barh(
            importance_df['Feature'],
            importance_df['Importance'],
            color=self.main_color,
            alpha=0.8
        )
        
        plt.xlabel('Mean |SHAP Value|', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title('Top {} Feature Importance (SHAP)'.format(max_display), fontsize=14, pad=15)
        
        plt.gca().invert_yaxis()
        sns.despine()
        plt.tight_layout()
        plt.savefig(full_path, dpi=600)
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
            self.plot_confusion_matrix()
            self.plot_classification_report()
            self.plot_feature_importance()
            
            print("\n✅ 完整分析完成！")
            
        finally:
            if old_dir:
                os.chdir(old_dir)


# ======================
# 🚀 使用示例
# ======================
if __name__ == "__main__":
    # 创建工具实例
    tool = SVMClassificationTool(
        file_path="test_classification_data.xlsx",
        multiple_folder=5
    )
    
    # 运行完整分析
    tool.run_full_analysis()
