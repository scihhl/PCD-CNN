import numpy as np
from data import test_extract

class ObjectHandler:
    def __init__(self, frame_data):
        """
        Initialize the ObjectHandler with frame data.
        :param frame_data: List of dictionaries containing frame data.
        """
        self.frame_data = frame_data
        self.poses = self.pose_sequence(frame_data)
        self.times = self.timestamp_sequence(frame_data)
        (self.camera_movement, self.camera_velocity, self.camera_acceleration,
         self.camera_angle_change, self.camera_angle_change_speed) = self.camera_movement_sequence()
        self.obj = []

    def object_sequence(self, target):
        """
        Create a sequence of objects for a specific target object ID across frames.
        :param target: The target object ID to track.
        """
        sequence = []
        for frame in self.frame_data:
            is_found = False
            for label in frame['labels']:
                if label['object_id'] == target:
                    sequence.append(label)
                    is_found = True
                    break
            if not is_found:
                sequence.append(None)
        self.obj = sequence

    @staticmethod
    def timestamp_sequence(frame_data):
        """
        Extract timestamps from frame data.
        :param frame_data: List of dictionaries containing frame data.
        :return: List of timestamps.
        """
        return [frame['pose']['timestamp'] for frame in frame_data]

    @staticmethod
    def pose_sequence(frame_data):
        """
        Extract pose positions from frame data.
        :param frame_data: List of dictionaries containing frame data.
        :return: List of pose positions.
        """
        return [frame['pose']['position'] for frame in frame_data]

    def camera_movement_sequence(self):
        """
        Compute camera movement, velocity, acceleration, angle change, and angle change speed.
        :return: Tuple containing camera movement, velocity, acceleration, angle change, and angle change speed.
        """
        camera_location_vectors = np.array([[point['x'], point['y'], point['z']] for point in self.poses])
        camera_time_vector = self.times
        return ObjectFeaturesExtractor.compute_movement(camera_location_vectors, camera_time_vector)

    @staticmethod
    def all_obj(frame_data):
        """
        Extract all unique object IDs from the frame data.
        :param frame_data: List of dictionaries containing frame data.
        :return: Set of unique object IDs.
        """
        unique_object_ids = set()

        # 遍历列表中的每个元素（每个元素都是一个字典）
        for item in frame_data:
            # 在每个字典中访问'labels'键，它的值是一个列表
            labels = item['labels']
            # 遍历'labels'列表中的每个元素（每个元素都是一个字典）
            for label in labels:
                # 在每个字典中访问'object_id'键，并将其添加到集合中
                object_id = label['object_id']
                unique_object_ids.add(object_id)

        # 打印所有不重复的object_id
        return unique_object_ids


class ObjectFeaturesExtractor:
    def __init__(self, handler: ObjectHandler):
        """
        Initialize the feature extractor with a given object handler.
        :param handler: An object handler that contains the necessary data.
        """
        self.handler = handler
        self.point_clouds = handler.obj
        self.poses = handler.poses
        self.timestamps = handler.times
        self.camera_movement, self.camera_velocity, self.camera_acceleration = (
            handler.camera_movement, handler.camera_velocity, handler.camera_acceleration)

    @staticmethod
    def compute_principal_axes(point_cloud):
        """
        Compute and return the principal components (PCA) directions of a point cloud.
        :param point_cloud: An Open3D point cloud object.
        :return: The principal axes (eigenvectors) of the point cloud.
        """
        points_matrix = np.asarray(point_cloud.points)
        mean = np.mean(points_matrix, axis=0)
        cov_matrix = np.cov((points_matrix - mean).T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        sort_indices = np.argsort(eigenvalues)[::-1]
        return eigenvectors[:, sort_indices]

    @staticmethod
    def compute_bounding_box(point_cloud):
        """
        Compute the bounding box dimensions of a point cloud.
        :param point_cloud: An Open3D point cloud object.
        :return: The dimensions of the oriented bounding box.
        """
        obb = point_cloud.get_oriented_bounding_box()
        return obb.extent

    @staticmethod
    def compute_density(point_cloud, volume):
        """
        Compute the density of a point cloud.
        :param point_cloud: The point cloud object.
        :param volume: The volume of the bounding box.
        :return: The density of the point cloud (points per unit volume).
        """
        points = np.asarray(point_cloud.points).shape[0]
        return points / volume if volume != 0 else 0

    @staticmethod
    def compute_diff(vector):
        """
        Compute the difference between consecutive elements of a vector.
        :param vector: The input vector.
        :return: A new vector of the same length containing the differences.
        """
        diff_vectors = np.diff(vector, axis=0)
        # 创建一个新的数组，第一个位置用第一个差分向量填充
        extended_diff_vectors = np.empty_like(vector)
        extended_diff_vectors[1:] = diff_vectors
        extended_diff_vectors[0] = diff_vectors[0]  # 假设第一个差分与第一个计算的差分相同
        return extended_diff_vectors

    @staticmethod
    def compute_timediff(timestamps):
        """
        Compute the time differences between consecutive timestamps.
        :param timestamps: The input vector of timestamps.
        :return: A new vector containing the time differences.
        """
        time_intervals = np.diff(timestamps)
        extended_time_intervals = np.insert(time_intervals, 0, time_intervals[0])
        return extended_time_intervals

    @staticmethod
    def compute_velocity(position_diff_vectors, extended_time_intervals):
        """
        Compute the velocity vectors based on position differences and time intervals.
        :param position_diff_vectors: The position difference vectors.
        :param extended_time_intervals: The time intervals.
        :return: The velocity vectors.
        """
        return position_diff_vectors / extended_time_intervals[:, np.newaxis]

    @staticmethod
    def compute_acceleration(velocity_diff_vectors, extended_time_intervals):
        """
        Compute the acceleration vectors based on velocity differences and time intervals.
        :param velocity_diff_vectors: The velocity difference vectors.
        :param extended_time_intervals: The time intervals.
        :return: The acceleration vectors.
        """
        return velocity_diff_vectors / extended_time_intervals[:, np.newaxis]

    @staticmethod
    def compute_cosine_similarity(vector1, vector2):
        """
        Compute the cosine similarity between two vectors.
        :param vector1: The first vector.
        :param vector2: The second vector.
        :return: The cosine similarity.
        """
        dot_product = np.dot(vector1, vector2)
        norms_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        cos_theta = dot_product / norms_product
        return cos_theta

    @staticmethod
    def compute_movement(location_vectors, time_vectors):
        """
        Compute the movement, velocity, and acceleration vectors, as well as angle changes and angle change speeds.
        :param location_vectors: The position vectors.
        :param time_vectors: The time vectors.
        :return: Movement vectors, velocity vectors, acceleration vectors, angle changes, and angle change speeds.
        """
        self = ObjectFeaturesExtractor
        movement = self.compute_diff(location_vectors)
        time_diff = self.compute_timediff(time_vectors)

        velocity = self.compute_velocity(movement, time_diff)
        velocity_diff = self.compute_diff(velocity)
        acceleration = self.compute_acceleration(velocity_diff, time_diff)

        angel_change = self.compute_angle_change(velocity)
        angel_change_speed = angel_change / time_diff

        return movement, velocity, acceleration, angel_change, angel_change_speed

    def extract_features(self):
        """
        Extract features from the point clouds and associated data.
        :return: A list of feature dictionaries for each object.
        """
        features = []

        label_sequence = []
        for i, label in enumerate(self.point_clouds):
            if label is None:
                continue
            label_sequence.append([i, label])

        (acceleration_magnitudes, obj_accelerations, obj_angle_changes,
         obj_angle_speeds, obj_movements, obj_velocities, velocity_magnitudes) = self.obj_movement_sequence(
            label_sequence)

        for label_id, (i, label) in enumerate(label_sequence):

            point_cloud = label['point_cloud']

            density, obb_extent, surface_area, volume = self.compute_size(point_cloud)

            density /= self.distance_between_points([self.poses[i], label['position']]) ** 2

            pca_z_cosine, principal_axis = self.compute_main_axis(point_cloud)

            height, length, width, xy_area = self.determine_3d_value(obb_extent, pca_z_cosine)

            (acceleration_magnitude, angle_change, angle_speed, obj_acceleration, obj_velocity,
             principal_camera_cosine, velocity_camera_cosine, velocity_magnitude) = self.compute_obj_motion(
                acceleration_magnitudes, i, label_id, obj_accelerations, obj_angle_changes, obj_angle_speeds,
                obj_movements, obj_velocities, principal_axis, velocity_magnitudes)


            feature_vector = {
                'length': length,
                'width': width,
                'height': height,
                'volume': volume,
                'surface_area': surface_area,
                'xy_area': xy_area,
                'density': density,
                'pca_z_cosine': pca_z_cosine,
                'principal_camera_cosine': principal_camera_cosine,
                'velocity_camera_cosine': velocity_camera_cosine,
                'velocity_magnitude': velocity_magnitude,
                'acceleration_magnitude': acceleration_magnitude,
                'object_type': label['object_type'],
                'angle_change': angle_change,
                'angle_speed': angle_speed,
                'location_x': label['position']['x'],
                'location_y': label['position']['y'],
                'location_z': label['position']['z'],
                'timestamp': self.timestamps[i],
                'object_id': label['object_id'],
                'principal_axis_x': principal_axis[0],
                'principal_axis_y': principal_axis[1],
                'principal_axis_z': principal_axis[2],

            }

            features.append(feature_vector)

        return features

    def compute_obj_motion(self, acceleration_magnitudes, i, label_id, obj_accelerations, obj_angle_changes,
                           obj_angle_speeds, obj_movements, obj_velocities, principal_axis, velocity_magnitudes):
        """
        Compute motion-related features for an object.
        :param acceleration_magnitudes: List of acceleration magnitudes for objects.
        :param i: Index of the current object.
        :param label_id: Label ID of the object.
        :param obj_accelerations: List of acceleration vectors for objects.
        :param obj_angle_changes: List of angle changes for objects.
        :param obj_angle_speeds: List of angle speeds for objects.
        :param obj_movements: List of movement vectors for objects.
        :param obj_velocities: List of velocity vectors for objects.
        :param principal_axis: Principal axis vector.
        :param velocity_magnitudes: List of velocity magnitudes for objects.
        :return: Tuple containing motion features of the object.
        """

        obj_movement, obj_velocity, obj_acceleration = (
            obj_movements[label_id], obj_velocities[label_id], obj_accelerations[label_id])

        (velocity_magnitude, acceleration_magnitude,
         angle_change, angle_speed) = (velocity_magnitudes[label_id], acceleration_magnitudes[label_id],
                                       obj_angle_changes[label_id], obj_angle_speeds[label_id])

        principal_camera_cosine = abs(self.compute_cosine_similarity(principal_axis, self.camera_velocity[i]))
        velocity_camera_cosine = abs(self.compute_cosine_similarity(obj_velocity, self.camera_velocity[i]))
        return (acceleration_magnitude, angle_change, angle_speed, obj_acceleration,
                obj_velocity, principal_camera_cosine, velocity_camera_cosine, velocity_magnitude)

    def obj_movement_sequence(self, label_sequence):
        """
        Compute the movement sequence of an object based on its label sequence.
        :param label_sequence: Sequence of labels for the object.
        :return: Tuple containing movement-related features of the object.
        """
        object_location = np.array([[label['position']['x'], label['position']['y'], label['position']['z']]
                                    for _, label in label_sequence])
        object_time = [self.timestamps[i] for i, _ in label_sequence]
        obj_movements, obj_velocities, obj_accelerations, obj_angle_changes, obj_angle_speeds = (
            self.compute_movement(object_location, object_time))

        velocity_magnitudes = np.linalg.norm(obj_velocities, axis=1)
        acceleration_magnitudes = np.linalg.norm(obj_accelerations, axis=1)

        return (acceleration_magnitudes, obj_accelerations, obj_angle_changes,
                obj_angle_speeds, obj_movements, obj_velocities, velocity_magnitudes)

    @staticmethod
    def determine_3d_value(obb_extent, pca_z_cosine):
        """
        Determine the 3D dimensions of an object based on its bounding box extent and PCA Z-axis cosine similarity.
        :param obb_extent: Extent of the oriented bounding box (length, width, height).
        :param pca_z_cosine: Cosine similarity between the principal axis and the Z-axis.
        :return: Tuple containing the height, length, width, and XY area of the object.
        """
        if pca_z_cosine >= 0.5:
            length, width, height = obb_extent[1], obb_extent[2], obb_extent[0]
        else:
            length, width, height = obb_extent[0], obb_extent[1], obb_extent[2]
        xy_area = length * width
        return height, length, width, xy_area

    @staticmethod
    def distance_between_points(points):
        """
        Calculate the Euclidean distance between two points.
        :param points: List containing two points, each represented as a dictionary or array.
        :return: Euclidean distance between the points.
        """
        points = [[point["x"], point["y"], point["z"]] if type(point) is dict else point for point in points]

        # 将点的坐标转换为numpy数组
        point1 = np.array(points[0])
        point2 = np.array(points[1])

        # 计算两点之间的欧氏距离
        distance = np.linalg.norm(point1 - point2)

        return distance

    @staticmethod
    def compute_main_axis(point_cloud):
        """
        Compute the principal axis of a point cloud using PCA.
        :param point_cloud: Open3D point cloud object.
        :return: Tuple containing the cosine similarity of the principal axis with the Z-axis and the principal axis vector.
        """
        # PCA 主轴
        self = ObjectFeaturesExtractor
        pca_axes = self.compute_principal_axes(point_cloud)
        z_axis = np.array([0, 0, 1])
        principal_axis = pca_axes[:, 0]
        # 主轴与Z轴的余弦相似度
        pca_z_cosine = abs(self.compute_cosine_similarity(principal_axis, z_axis))
        return pca_z_cosine, principal_axis

    @staticmethod
    def compute_size(point_cloud):
        """
        Compute the size-related features of a point cloud.
        :param point_cloud: Open3D point cloud object.
        :return: Tuple containing density, bounding box extent, surface area, and volume of the point cloud.
        """

        self = ObjectFeaturesExtractor
        obb_extent = self.compute_bounding_box(point_cloud)
        volume = np.prod(obb_extent)
        surface_area = 2 * (
                obb_extent[0] * obb_extent[1] + obb_extent[1] * obb_extent[2] + obb_extent[2] * obb_extent[0])
        density = self.compute_density(point_cloud, volume)
        return density, obb_extent, surface_area, volume

    @staticmethod
    def compute_angle_change(velocity_vectors):
        """
        Compute the angle changes between consecutive velocity vectors.
        :param velocity_vectors: List of velocity vectors.
        :return: Array of angle changes.
        """
        angle_changes = []
        for i in range(1, len(velocity_vectors)):
            angle_change = ObjectFeaturesExtractor.compute_cosine_similarity(velocity_vectors[i - 1],
                                                                             velocity_vectors[i])
            angle_changes.append(angle_change)
        return np.array([angle_changes[0]] + angle_changes)  # 第一个时间点没有前一个速度，可以标记为0或其他适当的值


class StaticFeaturesExtractor:
    def __init__(self, dataset):
        """
        Initialize the static feature extractor.
        :param dataset: Structured data containing point clouds and other relevant information.
        """
        self.tools = ObjectFeaturesExtractor
        self.frame_data = dataset

    def extract_features(self):
        """
        Extract static features from all point clouds in the dataset.
        :return: A list of dictionaries, each containing features for one object.
        """
        features_list = []
        for data in self.frame_data:
            features = self.compute_static_features(data)
            features_list += features
        return features_list

    @staticmethod
    def compute_geometric_center(point_cloud):
        """
        Compute the geometric center of a point cloud.
        :param point_cloud: An Open3D point cloud object.
        :return: Coordinates of the geometric center (numpy.ndarray).
        """
        points = np.asarray(point_cloud.points)
        geometric_center = np.mean(points, axis=0)
        return geometric_center

    def compute_static_features(self, dataset):
        """
        Compute static features for each object in a dataset using its point cloud.
        :param dataset: A dataset containing the point clouds of single or multiple objects.
        :return: A list of feature dictionaries for each object.
        """
        # Skip objects with fewer than 5 points as they are too small to provide reliable features.
        features_list = []
        for obj in dataset['extract_obj']:
            if np.asarray(obj.points).shape[0] < 5:
                continue

            features = {}
            # Compute the principal axis using PCA and calculate the cosine of the angle with the Z-axis.
            pca_z_cosine, principal_axis = self.tools.compute_main_axis(obj)

            # Compute density, extent of the oriented bounding box, surface area, and volume.
            density, obb_extent, surface_area, volume = self.tools.compute_size(obj)

            center_location = self.compute_geometric_center(obj)
            camera_location = dataset['pose']['position']

            # Adjust density based on the squared distance from the camera to the object center.
            density /= self.tools.distance_between_points([camera_location, center_location]) ** 2

            # Determine the dimensions based on the bounding box extent and the principal axis angle.
            height, length, width, xy_area = self.tools.determine_3d_value(obb_extent, pca_z_cosine)

            # Populate features for this object.
            features['width'], features['height'], features['length'] = width, height, length
            features['volume'] = volume
            features['surface_area'] = surface_area
            features['xy_area'] = xy_area
            features['density'] = density
            features['principal_axis_x'] = principal_axis[0]
            features['principal_axis_y'] = principal_axis[1]
            features['principal_axis_z'] = principal_axis[2]
            features['pca_z_cosine'] = pca_z_cosine

            # Store the object's location and timestamp data.
            features['location_x'] = center_location[0]
            features['location_y'] = center_location[1]
            features['location_z'] = center_location[2]
            features['timestamp'] = dataset['pose']['timestamp']

            # Store the camera's location and quaternion for pose orientation.
            features['camera_location_x'] = dataset['pose']['position']['x']
            features['camera_location_y'] = dataset['pose']['position']['y']
            features['camera_location_z'] = dataset['pose']['position']['z']
            features['camera_quaternion_x'] = dataset['pose']['quaternion']['x']
            features['camera_quaternion_y'] = dataset['pose']['quaternion']['y']
            features['camera_quaternion_z'] = dataset['pose']['quaternion']['z']
            features['camera_quaternion_w'] = dataset['pose']['quaternion']['w']

            features_list.append(features)

        return features_list


if __name__ == '__main__':
    frame_data = test_extract()
    static_extractor = StaticFeaturesExtractor(frame_data)
    static_extractor.extract_features()
