import numpy as np
import trimesh
from scipy.spatial.distance import euclidean
from track import ObjectTracker

class EvaluationMetrics:
    def __init__(self, df_pred, df_true, iou_threshold=0.1):
        """
        Initialize the EvaluationMetrics class with predicted and true dataframes.
        :param df_pred: DataFrame containing predicted object data.
        :param df_true: DataFrame containing true object data.
        :param iou_threshold: Threshold for considering a match valid based on Intersection over Union (IoU).
        """
        self.df_pred = df_pred.sort_values(by='timestamp').reset_index(drop=True)
        self.df_true = df_true.sort_values(by='timestamp').reset_index(drop=True)
        self.iou_threshold = iou_threshold  # Threshold for Intersection over Union (IoU)

    @staticmethod
    def calculate_iou(bbox_pred, bbox_true):
        """
        Calculate the Intersection over Union (IoU) of two oriented bounding boxes.
        :param bbox_pred: Predicted bounding box.
        :param bbox_true: True bounding box.
        :return: IoU value.
        """
        # Calculate the volume of intersection
        intersection_mesh = bbox_pred.intersection(bbox_true)
        intersection_volume = intersection_mesh.volume if intersection_mesh.is_volume else 0

        # Calculate the volume of union
        union_volume = bbox_pred.volume + bbox_true.volume - intersection_volume

        # Calculate the IoU
        iou = intersection_volume / union_volume if union_volume > 0 else 0
        return iou

    @staticmethod
    def create_oriented_bbox(center, dimensions, angles):
        """
        Create an oriented bounding box based on center coordinates, dimensions, and rotation angles.
        :param center: Center coordinates of the bounding box.
        :param dimensions: Dimensions (length, width, height) of the bounding box.
        :param angles: Euler angles for rotation.
        :return: Oriented bounding box.
        """
        obb = trimesh.primitives.Box(extents=dimensions,
                                     transform=trimesh.transformations.euler_matrix(angles[0], angles[1], angles[2]))
        obb.apply_translation(center)
        return obb

    def evaluate(self):
        """
        Evaluate the predictions against the true data based on position error and IoU.
        :return: Dictionary of evaluation metrics including Mean Position Error, Mean IoU, Precision, and Recall.
        """
        # Group predictions and truths by timestamps
        grouped_pred = self.df_pred.groupby('timestamp')
        grouped_true = self.df_true.groupby('timestamp')
        results = {'position_error': [], 'iou': []}
        matched_true_objects = set()

        # Intersect timestamps from predictions and truths
        timestamps = set(grouped_pred.groups.keys()) & set(grouped_true.groups.keys())

        for timestamp in timestamps:
            df_pred_timestamp = grouped_pred.get_group(timestamp)
            df_true_timestamp = grouped_true.get_group(timestamp)
            n = df_pred_timestamp.iloc[0].name
            closest_indices, min_distances = self.find_closest_matches(df_pred_timestamp, df_true_timestamp)

            for idx, pred_obj in df_pred_timestamp.iterrows():
                closest_idx = closest_indices[idx-n]
                closest_obj = df_true_timestamp.iloc[closest_idx]
                pred_bbox = self.create_oriented_bbox(pred_obj[['location_x', 'location_y', 'location_z']].values,
                                                      pred_obj[['length', 'width', 'height']].values,
                                                      pred_obj[['principal_axis_x', 'principal_axis_y',
                                                                'principal_axis_z']].values)
                true_bbox = self.create_oriented_bbox(closest_obj[['location_x', 'location_y', 'location_z']].values,
                                                      closest_obj[['length', 'width', 'height']].values,
                                                      closest_obj[['principal_axis_x', 'principal_axis_y',
                                                                   'principal_axis_z']].values)

                iou = self.calculate_iou(pred_bbox, true_bbox)
                pos_error = euclidean(pred_obj[['location_x', 'location_y', 'location_z']].values,
                                      closest_obj[['location_x', 'location_y', 'location_z']].values)
                results['iou'].append(iou)
                if iou >= self.iou_threshold:
                    results['position_error'].append(pos_error)
                    matched_true_objects.add(closest_obj.name)

        metrics = {
            'Mean Position Error': np.mean(results['position_error']),
            'Mean IoU': np.mean(results['iou']),
            'Precision': len(matched_true_objects) / self.df_pred['object_id'].size if self.df_pred['object_id'].size > 0 else 0,
            'Recall': len(matched_true_objects) / self.df_true['object_id'].size if self.df_true['object_id'].size > 0 else 0
        }
        return metrics

    @staticmethod
    def find_closest_matches(df_pred, df_true):
        """
        Find closest matches between predicted and true objects based on Euclidean distance.
        :param df_pred: DataFrame of predicted objects.
        :param df_true: DataFrame of true objects.
        :return: Tuple of closest indices and minimum distances.
        """
        distances = ObjectTracker.calculate_euclidean_distance(df_pred, df_true)
        closest_indices = np.argmin(distances, axis=1)
        min_distances = np.min(distances, axis=1)
        return closest_indices, min_distances
