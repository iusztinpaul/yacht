from typing import List

import pandas as pd
import matplotlib.pyplot as plt


class BaseRenderer:
    def show(self):
        plt.show()

    def time_series(self, data: pd.DataFrame, features: List[str], asset_index_name: str):
        assets = data.index.get_level_values(asset_index_name).unique()
        data = data.unstack(level=asset_index_name)

        fig, axes = plt.subplots(len(features), 1, figsize=(15, 10))
        for feature_ax_idx, feature in enumerate(features):
            axes[feature_ax_idx].set_ylabel(feature)
            axes[feature_ax_idx].set_xlabel('Time')

            for asset in assets:
                plotting_data = data[(feature, asset)]

                plotting_data.plot(ax=axes[feature_ax_idx])

        axes[0].legend()
