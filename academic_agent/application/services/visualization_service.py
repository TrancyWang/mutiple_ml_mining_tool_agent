"""可视化应用服务。"""

from __future__ import annotations


class VisualizationService:
    @property
    def tool(self):
        from academic_agent.tools.text_mining_tools import text_mining_tools
        from academic_agent.tools.viz_tools import viz_tools

        viz_tools.set_data(text_mining_tools.current_data)
        return viz_tools

    def wordcloud(self, text_column=None, title="词云图"):
        return self.tool.generate_wordcloud(text_column, title)

    def cluster_plot(self, cluster_column=None, title="聚类分布"):
        return self.tool.plot_distribution(cluster_column, title)

    def sentiment_plot(self, sentiment_column=None, title="情感分布"):
        return self.tool.plot_distribution(sentiment_column, title)

    def line_chart(self, x_column, y_column, title=None):
        return self.tool.plot_line_chart(x_column, y_column, title)


visualization_service = VisualizationService()
