"""文本挖掘应用服务。"""

from __future__ import annotations


class TextMiningService:
    @property
    def tool(self):
        from academic_agent.tools.text_mining_tools import text_mining_tools

        return text_mining_tools

    def preprocess(self, text_column=None):
        return self.tool.preprocess_text(text_column)

    def sentiment(self, text_column=None, mode="chinese"):
        return self.tool.sentiment_analysis(text_column, mode)

    def clustering(self, text_column=None, n_clusters=5, algorithm="kmeans", eps=0.5, min_samples=5):
        return self.tool.text_clustering(
            text_column, int(n_clusters), algorithm, float(eps), int(min_samples)
        )

    def keywords(self, text_column=None, top_n=10):
        return self.tool.extract_keywords(text_column, int(top_n))


text_mining_service = TextMiningService()
