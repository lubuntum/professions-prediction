import json
from typing import List

from models.Pupil import Pupil


#По факту сервис извлекает все параметры которые разрешены в test_mapping
#тем самым мы сможем менять разрешенные тесты (сейчас 3) флаг is_active
class ParamsExtractorService:
    def __init__(self, config_path: str = "./test_mapping.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.test_mapping = self.config['test_mapping']
        self.default_value = self.config.get('default_value', 0.0)

        # Build simple lookup: param_name -> test_type
        self.param_to_test = {}
        for test_type, test_config in self.test_mapping.items():
            if test_config.get('is_active', False):  # Only active tests
                for param_name in test_config['param_names']:
                    self.param_to_test[param_name] = test_type

    def extract_features(self, pupil: Pupil, feature_cols: List[str]) -> List[float]:
        """Extract feature values in the order specified by feature_cols"""
        feature_values = []

        # Loop through each required feature
        for col in feature_cols:
            # Check if this parameter is from an active test
            if col not in self.param_to_test:
                feature_values.append(self.default_value)  # Not active -> default
                continue

            # Get which test this parameter belongs to
            test_type = self.param_to_test[col]

            # Check if pupil has this test
            if test_type not in pupil.psychTests:

                feature_values.append(self.default_value)  # Test missing -> default
                continue

            # Find the parameter value inside the test
            test = pupil.psychTests[test_type]
            param_value = next(
                (p.param for p in test.psychParams if p.name == col),
                self.default_value
            )
            feature_values.append(param_value)

        return feature_values