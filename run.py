import os
from data import PCDObjectExtractor, FrameDataProcessor, visualization, PointCloudManager
from feature import ObjectHandler, ObjectFeaturesExtractor, StaticFeaturesExtractor
import pandas as pd
from random_forest import RandomForestModel
import pickle
from track import ObjectTracker
from analyze import EvaluationMetrics
# Open3D for 3D data processing
# Citation:
# Zhou, Q.-Y., Park, J., & Koltun, V. (2018). Open3D: A Modern Library for 3D Data Processing. arXiv:1801.09847.


class Task:
    def __init__(self, base='data'):
        """
        Initialize the Task class.
        :param base: Base directory where data is stored.
        """
        self.base_dir = base
        self.train_set, self.test_set = self.split_dataset()
        self.train_data_path, self.test_data_path = 'temp/train_features.xlsx', 'temp/test_features.xlsx'
        self.static_model_path = 'temp/static_random_forest_model.pkl'
        self.full_model_path = 'temp/full_random_forest_model.pkl'
        self.static_feature_path = 'temp/static_test_features.xlsx'
        self.static_feature = ['length', 'width', 'height', 'volume', 'pca_z_cosine']
        self.full_feature = ['length', 'width', 'height', 'volume',
                             'pca_z_cosine', 'principal_camera_cosine', 'velocity_camera_cosine',
                             'velocity_magnitude', 'acceleration_magnitude', 'angle_speed']

    def prepare_data(self):
        """
        Extract features from the dataset and save them to Excel files.
        """
        self.extract_dataset_feature(self.train_set, filename=self.train_data_path)
        self.extract_dataset_feature(self.test_set, filename=self.test_data_path)

    def split_dataset(self):
        """
        Split the dataset into training and testing sets.
        :return: A tuple containing training and testing sets.
        """
        file_path = f"{self.base_dir}/tracking_train_label"
        directories = self.list_directories(file_path)
        train_set = directories[:-1]
        test_set = [directories[-1]]
        return train_set, test_set

    def prepare_static_feature(self):
        """
        Prepare static features from the test set and save them to an Excel file.
        """
        features = []
        frame_ids = []
        for directory in self.test_set[:1]:
            frame_id = directory
            features.append([])
            frame_ids.append(frame_id)
            processor = PCDObjectExtractor(self.base_dir, frame_id)
            frame_data = processor.extract_objects(eps=0.7, min_points=20)
            static_extractor = StaticFeaturesExtractor(frame_data)
            features_list = static_extractor.extract_features()
            features[-1] += features_list
        self.save_data(frame_ids, features, self.static_feature_path)

    def extract_dataset_feature(self, dataset, filename):
        """
        Extract features from the dataset and save them to an Excel file.
        :param dataset: List of directories containing the dataset.
        :param filename: Filename to save the extracted features.
        """
        features = []
        frame_ids = []
        for directory in dataset:
            frame_id = directory
            features.append([])
            frame_ids.append(frame_id)
            processor = FrameDataProcessor(self.base_dir, frame_id)
            frame_data, _ = processor.load_all_frame_data()
            targets = ObjectHandler.all_obj(frame_data)
            handler = ObjectHandler(frame_data)
            for target in targets:
                handler.object_sequence(target)
                if len(handler.obj) - handler.obj.count(None) > 1:
                    extractor = ObjectFeaturesExtractor(handler)
                    features[-1] += extractor.extract_features()
        self.save_data(frame_ids, features, filename)

    def generate_random_forest_model(self):
        """
        Generate and save RandomForest models for static and full features.
        """
        train_data = self.read_all_sheets(self.train_data_path)
        test_data = self.read_all_sheets(self.test_data_path)
        static = self.model_generator(train_data, test_data, self.static_feature, n_estimators=50, random_state=42)
        full = self.model_generator(train_data, test_data, self.full_feature, n_estimators=50, random_state=42)
        self.save_model(static, self.static_model_path)
        self.save_model(full, self.full_model_path)

    def tracking(self):
        """
        Perform object tracking using the trained RandomForest models and evaluate the results.
        """
        static_rf_model = self.load_model(self.static_model_path)  # 假设已经加载了训练好的模型
        full_rf_model = self.load_model(self.full_model_path)  # 假设已经加载了训练好的模型

        df = self.read_all_sheets(self.static_feature_path)
        static_tracker = ObjectTracker(static_rf_model, self.static_feature, df)
        static_tracker.update_tracking()
        df = static_tracker.compute_motion_features()
        df.reset_index(drop=True)

        full_tracker = ObjectTracker(full_rf_model, self.full_feature, df)
        res = full_tracker.df
        res.reset_index(drop=True)

        unique_sheets = res['SheetName'].unique()
        for sheet in unique_sheets:

            true = self.read_sheet(self.test_data_path, sheet)

            pred = res[res['SheetName'] == sheet].copy()
            pred.reset_index(drop=True)

            evaluator = EvaluationMetrics(pred, true)
            evaluate_metrics = evaluator.evaluate()
            print(evaluate_metrics)

            _, temp = FrameDataProcessor(self.base_dir, sheet).load_all_frame_data()

            for row in pred.itertuples():
                camera_position = [row.camera_location_x, row.camera_location_y, row.camera_location_z]
                camera_quaternion = [row.camera_quaternion_w, row.camera_quaternion_x, row.camera_quaternion_y,
                                     row.camera_quaternion_z]
                object_position = [row.location_x, row.location_y, row.location_z]
                object_size = [row.length, row.width, row.height]
                heading = PointCloudManager.calculate_heading_from_quaternion(camera_position, camera_quaternion,
                                                                              object_position)
                bbox = PointCloudManager.create_3d_bbox(object_position, object_size, heading)
                temp += [bbox]
            visualization(temp)

    @staticmethod
    def save_model(model, name):
        """
        Save a model to a file.
        :param model: The model to be saved.
        :param name: The filename to save the model.
        """
        with open(name, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load_model(name):
        """
        Load a model from a file.
        :param name: The filename to load the model from.
        :return: The loaded model.
        """
        with open(name, 'rb') as f:
            loaded_model = pickle.load(f)
        return loaded_model

    @staticmethod
    def save_data(sheet_names: list, features: list, name: str):
        """
        Save extracted features to an Excel file with multiple sheets.
        :param sheet_names: List of sheet names.
        :param features: List of features for each sheet.
        :param name: The filename to save the data.
        """
        if len(features) != len(sheet_names):
            raise ValueError("每个特征集必须有一个对应的页签名称")

        with pd.ExcelWriter(name, engine='openpyxl') as writer:
            for feature_list, sheet_name in zip(features, sheet_names):
                df = pd.DataFrame(feature_list)
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    @staticmethod
    def read_sheet(filename: str, sheet_name: str) -> pd.DataFrame:
        """
        Read a specific sheet from an Excel file.
        :param filename: The filename to read from.
        :param sheet_name: The name of the sheet to read.
        :return: DataFrame containing the data from the sheet.
        """
        df = pd.read_excel(filename, sheet_name=sheet_name)
        return df

    @staticmethod
    def read_all_sheets(filename: str) -> pd.DataFrame:
        """
        Read all sheets from an Excel file and concatenate them into a single DataFrame.
        :param filename: The filename to read from.
        :return: DataFrame containing data from all sheets.
        """
        xls = pd.ExcelFile(filename)
        all_data = pd.DataFrame()
        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name)
            df['SheetName'] = sheet_name
            all_data = pd.concat([all_data, df], ignore_index=True)
        return all_data

    @staticmethod
    def list_directories(path):
        """
        Return a list of all subdirectories in the given path.
        :param path: The directory path to list subdirectories.
        :return: List of subdirectory names.
        """
        entries = os.listdir(path)
        directories = [entry for entry in entries if os.path.isdir(os.path.join(path, entry))]
        return directories

    @staticmethod
    def model_generator(train_data, test_data, features, n_estimators=50, random_state=42):
        """
        Generate and train a RandomForest model.
        :param train_data: Training data.
        :param test_data: Testing data.
        :param features: List of feature column names.
        :param n_estimators: Number of trees in the forest.
        :param random_state: Seed for random number generator.
        :return: Trained RandomForest model.
        """
        model = RandomForestModel(features, 'object_type', n_estimators=n_estimators, random_state=random_state)
        model.input_data(train_data, test_data)
        model.train()
        model.evaluate()
        return model


if __name__ == '__main__':
    task = Task()
    task.prepare_data()
    task.prepare_static_feature()
    task.generate_random_forest_model()
    task.tracking()





