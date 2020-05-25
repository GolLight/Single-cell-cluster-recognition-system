import pytest

from sklearn.utils.estimator_checks import check_estimator

from consensuscluster import TemplateEstimator
from consensuscluster import TemplateClassifier
from consensuscluster import TemplateTransformer


@pytest.mark.parametrize(
    "Estimator", [TemplateEstimator, TemplateTransformer, TemplateClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
