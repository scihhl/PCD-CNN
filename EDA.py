import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

class PointCloudEDA:
    def __init__(self, info):
        """
        Initialize the EDA class with point cloud and bounding boxes data.
        :param info: Object containing point clouds and bounding boxes for different categories.
        """

        self.objects = self.merge_dict(info.objects_train, info.objects_test)
        self.bboxes = self.merge_dict(info.bboxes_train, info.bboxes_test)

    @staticmethod
    def merge_dict(dict1, dict2):
        merged_dict = {}
        for key in dict1:
            if key in dict2:
                merged_dict[key] = dict1[key] + dict2[key]
        return merged_dict

    def compute_statistics(self):
        """
        Compute statistics for each category of point clouds.
        Returns a dictionary containing the number of points, volumes, sizes, and densities for each category.
        """
        stats = {}
        for key in self.objects:
            point_clouds = self.objects[key]
            boxes = self.bboxes[key]
            num_points = np.array([len(pc.points) for pc in point_clouds])
            volumes = np.array([np.prod(obb.extent) for obb in boxes])  # Calculate volume from the extents
            sizes = np.array([obb.extent for obb in boxes])  # Use the extent directly
            densities = num_points / volumes

            stats[key] = {
                'num_points': num_points.tolist(),
                'volumes': volumes.tolist(),
                'sizes': sizes.tolist(),
                'densities': densities.tolist()
            }
        return stats

    @staticmethod
    def plot_histograms(stats, category, limit_percentile=None):
        """
        Plot histograms for a given category of data, optionally limiting the range to exclude outliers.
        :param stats: Statistics dictionary.
        :param category: Data category to plot ('num_points', 'volumes', 'densities').
        :param limit_percentile: Tuple (lower, upper) percentiles to define the range of the data to include in the histogram.
        """
        plt.figure(figsize=(10, 6))
        for key, stat in stats.items():
            data = np.array(stat[category])
            if limit_percentile:
                # Calculate percentile limits and filter data
                lower_limit = np.percentile(data, limit_percentile[0])
                upper_limit = np.percentile(data, limit_percentile[1])
                data = data[(data >= lower_limit) & (data <= upper_limit)]

            plt.hist(data, bins=20, alpha=0.5, label=f'Class {key}')

        plt.title(f'Histogram of {category}')
        plt.xlabel(category)
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()
    @staticmethod
    def plot_size_distribution_interactive(stats):
        """
        Create an interactive 3D scatter plot of the sizes (extent) for each category with the legend moved to the bottom right.
        """
        fig = go.Figure()
        for key in stats:
            sizes = np.array(stats[key]['sizes'])
            fig.add_trace(go.Scatter3d(
                x=sizes[:, 0],
                y=sizes[:, 1],
                z=sizes[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    opacity=0.5,
                ),
                name=f'Class {key}'
            ))

        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis_title='X extent',
                yaxis_title='Y extent',
                zaxis_title='Z extent'
            ),
            title='Interactive 3D Scatter Plot of Sizes',
            legend=dict(
                x=1,  # Position the legend to the far right of the plot area
                y=0,  # Position the legend at the bottom of the plot area
                orientation='h'  # Horizontal orientation of the legend items
            )
        )
        fig.show()

