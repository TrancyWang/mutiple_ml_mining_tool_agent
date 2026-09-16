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

    def repair_clustering(self, text_column=None):
        return self.tool.repair_text_clustering(text_column)

    def repair_sentiment(self, text_column=None):
        return self.tool.repair_sentiment_analysis(text_column)

    def keywords(self, text_column=None, top_n=10):
        return self.tool.extract_keywords(text_column, int(top_n))

    def entities(self, text_column=None, batch_size=8, max_texts=200, **kwargs):
        return self.tool.entity_recognition(
            text_column, int(batch_size), int(max_texts),
            provider=kwargs.get("provider"), model=kwargs.get("model"),
            base_url=kwargs.get("base_url"), api_key=kwargs.get("api_key"),
        )

    def relations(self, text_column=None, batch_size=8, max_texts=200, **kwargs):
        return self.tool.relation_extraction(
            text_column, int(batch_size), int(max_texts),
            provider=kwargs.get("provider"), model=kwargs.get("model"),
            base_url=kwargs.get("base_url"), api_key=kwargs.get("api_key"),
        )


text_mining_service = TextMiningService()
