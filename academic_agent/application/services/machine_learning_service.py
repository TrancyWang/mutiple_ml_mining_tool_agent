"""机器学习应用服务。"""

from __future__ import annotations


class MachineLearningService:
    @property
    def tool(self):
        from academic_agent.tools.ml_tools import ml_tools
        from academic_agent.tools.text_mining_tools import text_mining_tools

        ml_tools.set_data(text_mining_tools.current_data)
        return ml_tools

    def regression(self, target_var, feature_vars, model_type="linear", test_size=0.2, output_path=None):
        return self.tool.regression_analysis(target_var, feature_vars, model_type, float(test_size), output_path=output_path)

    def classification(self, target_var, feature_vars, model_type="svm", test_size=0.2, output_path=None):
        return self.tool.classification_analysis(target_var, feature_vars, model_type, float(test_size), output_path=output_path)

    def causal(self, treatment_var, outcome_var, control_vars, method="ols", output_path=None):
        return self.tool.causal_inference(treatment_var, outcome_var, control_vars, method, output_path=output_path)


machine_learning_service = MachineLearningService()

