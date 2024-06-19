import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from feature import ObjectFeaturesExtractor

class ObjectTracker:
    def __init__(self, model, feature, df):
        """
        Initialize the ObjectTracker with a machine learning model, feature set, and initial DataFrame.
        :param model: The predictive model used for classification or regression.
        :param feature: List of features to be used by the model.
        :param df: DataFrame containing the initial data.
        """
        self.next_object_id = 1
        self.object_appearance_count = {}
        self.model = model
        self.feature = feature
        self.df = df
        self.tools = ObjectFeaturesExtractor
        self.all_tracked_frames = []

        self.preprocess_data()
        if 'velocity_magnitude' not in feature:
            self.df['object_id'] = np.arange(len(self.df))
        self.count_id = [0 for _ in range(len(self.df))]

    @staticmethod
    def calculate_euclidean_distance(frame_n, frame_n_plus_1):
        """
        Calculate Euclidean distances between objects in two consecutive frames using their 3D coordinates.
        :param frame_n: DataFrame of current frame.
        :param frame_n_plus_1: DataFrame of the next frame.
        :return: A matrix of Euclidean distances.
        """

        loc_features = ['location_x', 'location_y', 'location_z']
        positions_n = frame_n[loc_features].values
        positions_n1 = frame_n_plus_1[loc_features].values

        diff = positions_n[:, np.newaxis, :] - positions_n1[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        return distances

    @staticmethod
    def calculate_cosine_similarities(features1, features2, weights):
        """
        Compute cosine similarities between features of two sets of objects, considering specified weights.
        :param features1: Feature matrix for the first set.
        :param features2: Feature matrix for the second set.
        :param weights: Weights for each feature.
        :return: A matrix of cosine similarities.
        """

        weighted_features1 = features1 * weights
        weighted_features2 = features2 * weights

        norm_features1 = weighted_features1 / np.linalg.norm(weighted_features1, axis=1, keepdims=True)
        norm_features2 = weighted_features2 / np.linalg.norm(weighted_features2, axis=1, keepdims=True)

        cosine_similarities = np.dot(norm_features1, norm_features2.T)  # 注意使用转置
        return cosine_similarities

    @staticmethod
    def standardize_features(dataframe, model_features):
        """
        Standardize features in the DataFrame using a StandardScaler.
        :param dataframe: DataFrame containing the features to standardize.
        :param model_features: List of features to standardize.
        :return: Modified DataFrame with standardized features.
        """
        scaler = StandardScaler()
        dataframe[model_features] = scaler.fit_transform(dataframe[model_features])
        return dataframe

    def build_cost_matrix(self, frame_n, frame_n_plus_1):
        """
        Construct a cost matrix for tracking objects between two frames using Euclidean distance and cosine similarities.
        :param frame_n: DataFrame of the current frame.
        :param frame_n_plus_1: DataFrame of the next frame.
        :return: Cost matrix for object tracking.
        """
        weights = self.get_feature_weights(self.model)  # Assuming this method returns feature weights

        # Standardize features
        frame_n = self.standardize_features(frame_n.copy(), self.feature)
        frame_n_plus_1 = self.standardize_features(frame_n_plus_1.copy(), self.feature)

        # Compute distances and similarities
        distances = self.calculate_euclidean_distance(frame_n, frame_n_plus_1)
        cosine_similarities = self.calculate_cosine_similarities(frame_n[self.feature], frame_n_plus_1[self.feature], weights)

        label_mismatch_penalty = 10 * (frame_n['label'].values[:, None] != frame_n_plus_1['label'].values[None, :])

        # Calculate the cost matrix
        cost_matrix = distances + (1 - cosine_similarities) * 5 + label_mismatch_penalty
        return cost_matrix

    def preprocess_data(self):
        """
        Preprocess the data by predicting labels and filtering out certain labels and object sizes.
        """
        self.df['label'] = self.model.predict(self.df[self.feature])
        self.df = self.df[self.df['label'] != 6]
        self.df = self.df[self.df['label'] != 5]
        self.df = self.df[self.df['length'] < 10]
        self.df = self.df.reset_index(drop=True)

    @staticmethod
    def get_feature_weights(rf_model):
        """
        Retrieve and normalize feature weights from a RandomForest model.
        :param rf_model: RandomForest model from which to extract feature importances.
        :return: Normalized feature weights.
        """
        feature_weights = rf_model.model.feature_importances_
        total_importance = sum(feature_weights)
        normalized_feature_weights = feature_weights / total_importance
        return normalized_feature_weights

    def update_tracking(self):
        """
        Update object tracking across frames, linking objects based on their calculated similarity cost.
        """

        df_sorted = self.df.sort_values(['SheetName', 'timestamp'])
        df_sorted = df_sorted.reset_index(drop=True)

        grouped_by_sheet = df_sorted.groupby('SheetName')

        for sheet_name, sheet_data in grouped_by_sheet:
            previous_frame = None
            x = 0
            for timestamp, current_frame in sheet_data.groupby('timestamp'):
                if previous_frame is not None:
                    n = previous_frame.shape[0]

                    cost_matrix = self.build_cost_matrix(previous_frame, current_frame)
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)


                    for row, col in zip(row_ind, col_ind):

                        if cost_matrix[row, col] < 10:
                            current_frame.at[x + n + col, 'object_id'] = previous_frame.at[x + row, 'object_id']
                            self.count_id[current_frame.at[x + n + col, 'object_id']] += 1
                    x += n
                previous_frame = current_frame
                self.df.update(current_frame)

        valid_ids = [i for i, v in enumerate(self.count_id) if v > 3]
        self.df = self.df[self.df['object_id'].isin(valid_ids)]
        self.df.reset_index(drop=True)

    def compute_motion_features(self):
        """
        Compute and append motion-related features such as velocity and acceleration to the DataFrame.
        """
        grouped_data = self.df.groupby('object_id')
        self.df['principal_camera_cosine'] = np.nan
        self.df['velocity_camera_cosine'] = np.nan
        self.df['velocity_magnitude'] = np.nan
        self.df['acceleration_magnitude'] = np.nan
        self.df['angle_change'] = np.nan
        self.df['angle_speed'] = np.nan

        for object_id, group in grouped_data:
            if len(group) > 1:
                timestamps = group['timestamp'].to_numpy()
                locations = group[['location_x', 'location_y', 'location_z']].to_numpy()
                camera_locations = group[['camera_location_x', 'camera_location_y', 'camera_location_z']].to_numpy()
                principal_axis = group[['principal_axis_x', 'principal_axis_y', 'principal_axis_z']].to_numpy()

                (movements, velocities, accelerations,
                 angle_changes, angle_speeds) = self.tools.compute_movement(locations, timestamps)

                velocity_magnitudes = np.linalg.norm(velocities, axis=1)
                acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)

                (camera_movements, camera_velocities, camera_accelerations,
                 camera_angle_changes, camera_angle_change_speeds) = (
                    self.tools.compute_movement(camera_locations, timestamps))

                idx = group.index
                for i in range(len(group)):
                    self.df.at[idx[i], 'principal_camera_cosine'] = \
                        abs(self.tools.compute_cosine_similarity(principal_axis[i], camera_velocities[i]))
                    self.df.at[idx[i], 'velocity_camera_cosine'] = \
                        abs(self.tools.compute_cosine_similarity(velocities[i], camera_velocities[i]))
                    self.df.at[idx[i], 'velocity_magnitude'] = velocity_magnitudes[i]
                    self.df.at[idx[i], 'acceleration_magnitude'] = acceleration_magnitudes[i]
                    self.df.at[idx[i], 'angle_change'] = angle_changes[i]
                    self.df.at[idx[i], 'angle_speed'] = angle_speeds[i]
        return self.df
