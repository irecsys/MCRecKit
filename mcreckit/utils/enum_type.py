# @Time : 2024/10/24
# @Author : Yong Zheng


from enum import Enum
from recbole.utils.enum_type import ModelType as RecBoleModelType


class MCModelType(Enum):
    GENERAL = RecBoleModelType.GENERAL.value
    SEQUENTIAL = RecBoleModelType.SEQUENTIAL.value
    CONTEXT = RecBoleModelType.CONTEXT.value
    KNOWLEDGE = RecBoleModelType.KNOWLEDGE.value
    TRADITIONAL = RecBoleModelType.TRADITIONAL.value
    DECISIONTREE = RecBoleModelType.DECISIONTREE.value
    MULTICRITERIA = 7


class CustomColumn(Enum):
    """Define all custom constant data here

    """
    CRITERIA_VECTOR = 'criteria_vector'
    RANKING_SCORE = 'ranking_score'
    PREDICTED_LABEL = 'predicted_label'


class CustomConstant(Enum):
    CHAIN_MODELS = ['DeepCriteriaChain']
