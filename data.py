import os
import numpy as np
import open3d as o3d


class PointCloudLoader:
    def __init__(self, base_dir):
        """
        Initialize a PointCloudLoader object.
        :param base_dir: The base directory where point cloud files are stored.
        """
        self.base_dir = base_dir

    def construct_file_path(self, track_id, frame_id):
        """
        Construct the full path for PCD files.
        :param track_id: The tracking dataset ID (1, 2, or 3).
        :param frame_id: The frame number, such as '9048_1_frame'.
        :return: A list of full file paths for PCD files.
        """
        file_path = f"{self.base_dir}/tracking_train_pcd_{track_id}/result_{frame_id}_frame"
        return [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.pcd')]

    @staticmethod
    def load_point_cloud(file_path):
        """
        Load a single PCD file.
        :param file_path: The full path to a PCD file.
        :return: The loaded point cloud object.
        """
        pcd = o3d.io.read_point_cloud(file_path)
        return pcd

    def load_all_point_clouds(self, track_id, frame_id):
        """
        Load all PCD files for a given tracking frame.
        :param track_id: The tracking dataset ID.
        :param frame_id: The frame number.
        :return: A list of point cloud objects.
        """
        file_paths = self.construct_file_path(track_id, frame_id)
        point_clouds = [self.load_point_cloud(fp) for fp in file_paths]
        return point_clouds

    @staticmethod
    def convert_to_numpy(point_clouds):
        """
        Convert point cloud objects to numpy arrays.
        :param point_clouds: A list of point cloud objects.
        :return: A list of numpy arrays where each array represents points in a point cloud.
        """
        return [np.asarray(point_clouds[i].points) for i in range(len(point_clouds))]


class CameraPoseLoader:
    def __init__(self, base_dir):
        """
        Initialize a CameraPoseLoader object.
        :param base_dir: The base directory where camera pose files are stored.
        """
        self.base_dir = base_dir

    def construct_file_path(self, frame_id, file_id):
        """
        Construct the full path for a camera pose file.
        :param frame_id: The frame number, such as '9048_1_frame'.
        :param file_id: The file number, such as 233.
        :return: The complete file path.
        """
        file_path = f"{self.base_dir}/tracking_train_pose/result_{frame_id}_frame/{file_id}_pose.txt"
        return file_path

    @staticmethod
    def read_pose_file(file_path):
        """
        Read a single camera pose file.
        :param file_path: The complete path to a camera pose file.
        :return: A dictionary containing the camera position and orientation data.
        """
        with open(file_path, 'r') as file:
            data = file.readline().strip().split()
            pose_data = {
                'timestamp': float(data[1]),
                'position': {'x': float(data[2]), 'y': float(data[3]), 'z': float(data[4])},
                'quaternion': {'x': float(data[5]), 'y': float(data[6]), 'z': float(data[7]), 'w': float(data[8])}
            }
        return pose_data

    def load_camera_poses(self, frame_id, file_id):
        """
        Load specified camera pose information.
        :param frame_id: The frame number.
        :param file_id: The file number.
        :return: The camera position and orientation data.
        """
        file_path = self.construct_file_path(frame_id, file_id)
        return self.read_pose_file(file_path)


class ObjectLabelLoader:
    def __init__(self, base_dir):
        """
        Initialize an ObjectLabelLoader object.
        :param base_dir: The base directory where label files are stored.
        """
        self.base_dir = base_dir

    def construct_file_path(self, frame_id, file_id):
        """
        Construct the full path for a label file.
        :param frame_id: The frame number, such as '9048_1'.
        :param file_id: The file number, such as 233.
        :return: The complete file path.
        """
        file_path = f"{self.base_dir}/tracking_train_label/{frame_id}/{file_id}.txt"
        return file_path

    @staticmethod
    def read_label_file(file_path):
        """
        Read a single label file.
        :param file_path: The complete path to a label file.
        :return: A list containing all object label data.
        """
        objects = []
        with open(file_path, 'r') as file:
            for line in file:
                data = line.strip().split()
                label_data = {
                    'object_id': int(data[0]),
                    'object_type': int(data[1]),
                    'position': {'x': float(data[2]), 'y': float(data[3]), 'z': float(data[4])},
                    'size': {'length': float(data[5]), 'width': float(data[6]), 'height': float(data[7])},
                    'heading': float(data[8])
                }
                objects.append(label_data)
        return objects

    def load_object_labels(self, frame_id, file_id):
        """
        Load specified label information.
        :param frame_id: The frame number.
        :param file_id: The file number.
        :return: A list containing all object labels.
        """
        file_path = self.construct_file_path(frame_id, file_id)
        return self.read_label_file(file_path)

    @staticmethod
    def classification_objects(objs, target):
        """
        Filter objects by specified category.
        :param objs: List of object dictionaries.
        :param target: Target category, where 1 = small vehicles, 2 = large vehicles, 3 = pedestrians,
         4 = motorcyclists and bicyclists, 5 = traffic cones, and 6 = others.
        :return: A list containing objects of the target category.
        """
        return [obj for obj in objs if objs['object_type'] == target]


class FrameDataProcessor:
    def __init__(self, base_dir, frame_id):
        """
        Initialize a FrameDataProcessor object.
        :param base_dir: The root directory for the data.
        :param frame_id: The ID of the frame being processed, e.g., '9048_1_frame'.
        """
        self.base_dir = base_dir
        self.frame_id = frame_id
        self.pcd_loader = PointCloudLoader(base_dir)
        self.pose_loader = CameraPoseLoader(base_dir)
        self.label_loader = ObjectLabelLoader(base_dir)

    def load_frame_data(self, file_id):
        """
        Load all relevant data for a frame.
        :param file_id: The file number, e.g., '233'.
        :return: A dictionary containing pose, labels, and point clouds; returns None if an error occurs during loading.
        """
        try:
            # Load Pose data
            pose = self.pose_loader.load_camera_poses(self.frame_id, file_id)

            # Load Label data
            labels = self.label_loader.load_object_labels(self.frame_id, file_id)

            # Determine the track_id folder where the PCD file resides
            track_id = self.find_track_id()
            if track_id is None:
                raise ValueError("Frame ID not found in any track ID folder")

            # Load the PCD file corresponding to file_id
            pcd_path = os.path.join(self.base_dir, f'tracking_train_pcd_{track_id}',
                                    f'result_{self.frame_id}_frame', f'{file_id}.pcd')

            if not os.path.exists(pcd_path):
                raise FileNotFoundError(f"No PCD file found for file ID {file_id} in frame {self.frame_id}")

            point_cloud = self.pcd_loader.load_point_cloud(pcd_path)

            return {
                'pose': pose,
                'labels': labels,
                'point_cloud': point_cloud  # Return a single point cloud object
            }

        except Exception as e:  # Catch all possible exceptions
            print(f"An unexpected error occurred: {e}")
            return None


    def find_track_id(self):
        """
        Determine under which track_id the frame_id is located.
        """
        track_ids = [1, 2, 3]
        for track_id in track_ids:
            pcd_path = os.path.join(self.base_dir, f'tracking_train_pcd_{track_id}', f'result_{self.frame_id}_frame')
            if os.path.exists(pcd_path):
                return track_id
        return None

    def load_all_frame_data(self):
        """
        Load data for all frames, sort the file IDs numerically, and process each frame's data.
        """
        label_path = os.path.join(self.base_dir, 'tracking_train_label', self.frame_id)
        file_ids = [f.split('_')[0][:-4] for f in os.listdir(label_path) if f.endswith('.txt')]
        all_data = []

        for file_id in self.numeric_sort_key(file_ids):
            frame_data = self.load_frame_data(file_id)
            if frame_data is None:
                continue
            self.collect_object_points(frame_data)
            frame_data = self.traverse_points(frame_data, file_id)
            all_data.append(frame_data)
        temp = []
        for frame_data in all_data:
            temp += [frame_data['point_cloud']]
        return all_data, temp

    @staticmethod
    def numeric_sort_key(ids):
        """
        Extract numeric part from filenames for sorting.
        """
        int_numbers = [int(num) for num in ids]
        sorted_int_numbers = sorted(int_numbers)
        return [str(num) for num in sorted_int_numbers]

    @staticmethod
    def collect_object_points(frame_data):
        """
        Extract point clouds for objects (before coordinate transformation).
        """
        pc_manager = PointCloudManager(frame_data['point_cloud'])
        # 提取对象的点云（在坐标转换之前）
        pc_manager.extract_object_point_clouds(frame_data['labels'])

    def traverse_points(self, frame_data, file_id):
        """
        Transform entire point cloud to global coordinates and process each label for object point clouds,
        coordinates, heading, and bounding box.
        """
        try:
            # Convert the entire point cloud to global coordinates
            point_cloud_abs = self.transform_to_global_coordinates(frame_data['pose'],
                                                                   np.asarray(frame_data['point_cloud'].points))
            frame_data['point_cloud'].points = o3d.utility.Vector3dVector(point_cloud_abs)

            # Process each label for object point cloud, coordinates, heading, and bounding box
            for label in frame_data['labels']:
                # Convert object point cloud to global coordinates
                object_point_cloud = label['point_cloud']
                global_object_points = self.transform_to_global_coordinates(frame_data['pose'],
                                                                            np.asarray(object_point_cloud.points))
                object_point_cloud.points = o3d.utility.Vector3dVector(global_object_points)

                # Convert label position to global coordinates
                pos = np.array([[label['position']['x'], label['position']['y'], label['position']['z']]])
                transformed_pos = self.transform_to_global_coordinates(frame_data['pose'], pos)
                label['position']['x'], label['position']['y'], label['position']['z'] = transformed_pos[0]

                # Rotate heading
                label['heading'] = self.rotate_heading(frame_data['pose']['quaternion'], label['heading'])

                # Transform bbox object to global coordinates (if it exists)
                if 'bbox' in label:
                    bbox = label['bbox']
                    quaternion_array = np.array([frame_data['pose']['quaternion']['w'],
                                                 frame_data['pose']['quaternion']['x'],
                                                 frame_data['pose']['quaternion']['y'],
                                                 frame_data['pose']['quaternion']['z']]).reshape(4, 1)
                    global_rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion_array)
                    bbox_center_global = self.transform_to_global_coordinates(frame_data['pose'],
                                                                              np.array([bbox.center]))
                    bbox.center = bbox_center_global[0]
                    bbox.R = global_rotation_matrix @ bbox.R

            return frame_data

        except FileNotFoundError as e:
            print(f"Error loading data for file ID {file_id}: {e}")

    @staticmethod
    def transform_to_global_coordinates(pose, points):
        """
        Transform points from the local coordinate system to the global coordinate system based on the provided pose.
        :param pose: A dictionary containing the position and quaternion (orientation) of the pose.
        :param points: An array of points in the local coordinate system to be transformed.
        :return: An array of points transformed into the global coordinate system.
        """
        # 提取位置和四元数
        translation = np.array([pose['position']['x'], pose['position']['y'], pose['position']['z']])
        quaternion = np.array(
            [pose['quaternion']['w'], pose['quaternion']['x'], pose['quaternion']['y'], pose['quaternion']['z']])

        # 创建旋转矩阵
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion)

        # 应用旋转和平移
        points_transformed = np.dot(points, rotation_matrix.T) + translation
        return points_transformed

    @staticmethod
    def rotate_heading(quaternion, heading):
        """
        Rotate a 2D heading angle using a quaternion to convert it from a local coordinate system to a global coordinate system.
        :param quaternion: A dictionary containing the quaternion elements (w, x, y, z) representing the orientation.
        :param heading: A heading angle in radians in the local coordinate system (assuming 0 radians points north and increases counterclockwise).
        :return: The heading angle in radians in the global coordinate system.
        """
        # Convert the quaternion from the dictionary into a numpy array
        quaternion = np.array(
            [quaternion['w'], quaternion['x'], quaternion['y'], quaternion['z']])

        # Convert the heading angle to a 3D vector (ignoring z-axis because heading is in 2D)
        heading_vector = np.array([np.cos(heading), np.sin(heading), 0])

        # Calculate the rotation matrix from the quaternion
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion)

        # Rotate the heading vector using the rotation matrix
        global_heading_vector = np.dot(rotation_matrix, heading_vector)

        # Calculate the global heading angle from the rotated vector
        global_heading = np.arctan2(global_heading_vector[1], global_heading_vector[0])
        return global_heading


class PointCloudManager:
    def __init__(self, point_cloud):
        """
        Initialize the PointCloudManager with a point cloud.
        :param point_cloud: The Open3D point cloud object to be managed.
        """
        self.point_cloud = point_cloud

    @staticmethod
    def create_3d_bbox(position, size, heading):
        """
        Create an OrientedBoundingBox for an object.
        :param position: The global position of the object's center (x, y, z).
        :param size: The dimensions of the object as a dictionary (length, width, height).
        :param heading: The heading angle of the object around the z-axis in radians.
        :return: An OrientedBoundingBox object.
        """
        if isinstance(position, dict):
            center = np.array([position['x'], position['y'], position['z']])
        else:
            center = np.array(position)
        # If size is a dictionary, convert it to an array and divide by 2
        if isinstance(size, dict):
            extent = np.array([size['length'], size['width'], size['height']])
        else:
            extent = np.array(size)  # Assuming size is already in an array/list form

        # Compute the rotation matrix from the heading angle
        R = np.array([
            [np.cos(heading), -np.sin(heading), 0],
            [np.sin(heading), np.cos(heading), 0],
            [0, 0, 1]
        ])

        # Create an oriented bounding box with the correct orientation and extents
        bbox = o3d.geometry.OrientedBoundingBox(center, R, extent)
        return bbox

    def filter_points_in_bbox(self, bbox):
        """
        Filter points that are within the bounding box.
        :param bbox: An OrientedBoundingBox object.
        :return: A point cloud subset that lies within the bbox.
        """
        # Use the bounding box to crop the point cloud
        cropped_point_cloud = self.point_cloud.crop(bbox)
        return cropped_point_cloud

    def extract_object_point_clouds(self, labels):
        """
        Extract point clouds for each object based on its bounding box.
        :param labels: A list of label dictionaries containing position, size, and heading.
        """
        bboxes = []

        for label in labels:
            bbox = self.create_3d_bbox(label['position'], label['size'], label['heading'])
            bboxes.append(bbox)

            object_points = self.filter_points_in_bbox(bbox)
            # Ensure to convert the cropped point cloud back to Open3D PointCloud if necessary
            label['point_cloud'] = object_points if isinstance(object_points,
                                                               o3d.geometry.PointCloud) else o3d.geometry.PointCloud(
                object_points)
            label['bbox'] = bbox

    @staticmethod
    def quaternion_to_rotation_matrix(quaternion):
        """
        Convert a quaternion to a rotation matrix.
        :param quaternion: A numpy array representing the quaternion [w, x, y, z].
        :return: A 3x3 rotation matrix.
        """
        w, x, y, z = quaternion
        return np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
        ])

    @staticmethod
    def calculate_heading_from_quaternion(camera_position, camera_quaternion, object_position):
        """
        Calculate the heading angle from the camera's quaternion.
        :param camera_position: The position of the camera (x, y, z).
        :param camera_quaternion: The quaternion of the camera [w, x, y, z].
        :param object_position: The position of the object (x, y, z).
        :return: The heading angle in radians.
        """
        # Transform object position into camera coordinate system
        cam_rot_matrix = PointCloudManager.quaternion_to_rotation_matrix(camera_quaternion)
        relative_position = np.array(object_position) - np.array(camera_position)
        transformed_position = cam_rot_matrix.dot(relative_position[:3])

        # Calculate angle in 2D plane
        angle = np.arctan2(transformed_position[1], transformed_position[0])
        return angle


class PCDObjectExtractor(FrameDataProcessor):
    def __init__(self, base_dir, frame_id):
        """
        Initialize the PCDObjectExtractor which is a subclass of FrameDataProcessor.
        :param base_dir: The base directory where data is stored.
        :param frame_id: The specific frame ID to process.
        """
        super().__init__(base_dir, frame_id)

    def extract_objects(self, eps=0.2, min_points=10, z_threshold=0.3):
        """
        Extract all objects and their point cloud data from a specified frame using DBSCAN clustering.
        :param eps: The neighborhood radius for DBSCAN clustering.
        :param min_points: The minimum number of points required to form a cluster.
        :param z_threshold: The height threshold to filter out points based on the Z-axis.
        :return: A list containing point clouds of the extracted objects.
        """
        frame_data, _ = self.load_all_frame_data()

        # Calculate the 30th percentile of Z-axis values across all point cloud data in the frame
        all_z_values = np.concatenate([np.asarray(data['point_cloud'].points)[:, 2] for data in frame_data])
        percentile_z_value = np.percentile(all_z_values, 30)

        # Compute the absolute Z-axis threshold
        absolute_z_threshold = percentile_z_value + z_threshold

        for data in frame_data:
            point_cloud = data['point_cloud']
            points = np.asarray(point_cloud.points)

            #visualization([point_cloud])

            # Filter points below the absolute Z-axis threshold
            filtered_points = points[points[:, 2] > absolute_z_threshold]
            filtered_pcd = o3d.geometry.PointCloud()
            filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
            #visualization([filtered_pcd])

            # Perform DBSCAN clustering on filtered points
            labels = np.array(filtered_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

            # Extract clusters
            max_label = labels.max()
            extract_obj = []
            for i in range(max_label + 1):
                cluster_points = filtered_points[labels == i]
                object_pcd = o3d.geometry.PointCloud()
                object_pcd.points = o3d.utility.Vector3dVector(cluster_points)
                extract_obj.append(object_pcd)

                #obb = object_pcd.get_axis_aligned_bounding_box()
                #extract_obj.append(obb)
                #visualization(extract_obj)
            data['extract_obj'] = extract_obj

        return frame_data


def visualization(objs):
    o3d.visualization.draw_geometries(objs)
    return


def test_PCD():
    loader = PointCloudLoader(base_dir='data')
    point_clouds = loader.load_all_point_clouds(track_id=1, frame_id='9048_1')
    point_cloud_np = loader.convert_to_numpy(point_clouds)
    print(point_cloud_np)


def test_pose():
    pose_loader = CameraPoseLoader(base_dir='data')
    camera_pose = pose_loader.load_camera_poses(frame_id='9048_1', file_id='233')
    print(camera_pose)


def test_label():
    label_loader = ObjectLabelLoader(base_dir='data')
    object_labels = label_loader.load_object_labels(frame_id='9048_1', file_id='233')
    print(object_labels)


def test_frame():
    base_dir = 'data'
    frame_id = '9061_1'

    processor = FrameDataProcessor(base_dir, frame_id)
    frame_data, temp = processor.load_all_frame_data()

    for i, frame in enumerate(frame_data):
        print(f'for No.{i} frame we have:')
        print('Pose of this PCD files is', frame['pose'])
        print('Labels found in this PCD files are', len(frame['labels']))
        print('First label coordinates are', frame['labels'])
        print('PCD file is composed of', frame['point_cloud'])
        print('First point coordinates are', np.asarray(frame['point_cloud'].points)[0])
        print('+++++++++++++++++end++++++++++++++++')


def test_extract():
    base_dir = 'data'
    extractor = PCDObjectExtractor(base_dir, frame_id='9048_1')
    frame_data = extractor.extract_objects()
    return frame_data

# 测试
if __name__ == '__main__':
    # test_PCD()
    # test_pose()
    # test_label()
    # test_frame()
    test_extract()
