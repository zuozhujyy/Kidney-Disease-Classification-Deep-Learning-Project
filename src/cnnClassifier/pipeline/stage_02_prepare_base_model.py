from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier import logger


STAGE_NAME = "Prepare base model"

class PrepareBaseModelTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()

if __name__ == "__main__":
    try:
        logger.info("*" * 20)
        logger.info(f"{'>'*6} stage {STAGE_NAME} started {'<'*6}")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f"{'>'*6} stage {STAGE_NAME} completed {'<'*6}\n\n")
    except Exception as e:
        logger.exception(e)
        raise e