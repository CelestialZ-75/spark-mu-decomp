import numpy as np

from mu_course.prepare_cascade_prediction import impute_training_medians, raster_features


def test_track_features_and_missing_history():
    pulses = [np.array([100, 200, 300, 400]), np.array([], dtype=int)]
    x = raster_features(pulses, 1000, 1000, np.array([100, 300]), np.array([300, 500]), 20)
    assert x.shape == (2, 12)
    np.testing.assert_array_equal(x[:, 1], [10, 10])
    np.testing.assert_allclose(x[:, 3], [np.nan, 0], equal_nan=True)
    np.testing.assert_allclose(x[:, 4], [0.099, 0.099])
    np.testing.assert_array_equal(x[:, 5], [1, 1])
    np.testing.assert_array_equal(x[:, 7], [0, 0])
    assert np.isnan(x[:, 9]).all()


def test_missing_values_use_training_only():
    train = np.array([[1, np.nan], [3, np.nan], [np.nan, np.nan]])
    test = np.array([[100, 8], [np.nan, np.nan]])
    train_filled, test_filled, fill = impute_training_medians(train, test)
    np.testing.assert_array_equal(fill, [2, 0])
    np.testing.assert_array_equal(train_filled[-1], [2, 0])
    np.testing.assert_array_equal(test_filled[-1], [2, 0])
